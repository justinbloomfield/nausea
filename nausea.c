#include <err.h>
#include <complex.h>
#include <curses.h>
#include <fcntl.h>
#include <locale.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <wchar.h>

#include <fftw3.h>

#define LEN(x) (sizeof (x) / sizeof *(x))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define WCSLEN(s) (LEN(s) - 2)

#include "config.h"

static unsigned msec = 1000 / 25; /* 25 fps */
static unsigned nsamples = 44100 * 2; /* stereo */
static unsigned dftlen = 8192;
static unsigned dftout = 8192 / 2 + 1;
static wchar_t chbar = CHBAR;
static wchar_t chpeak = CHPEAK;
static wchar_t chpoint = CHPOINT;
static wchar_t intensity[] = INTENSITY;
static char *fname = "/tmp/audio.fifo";
static char *argv0;
static int colors;
static int peaks;
static int keep;
static int left;
static int bounce;
static int die;
static int freeze;
static int stereo;

struct frame {
	int fd;
	size_t width, width_old;
	size_t height, height_old;
	int *peak;
	int *sav;
#define PK_HIDDEN -1
	int16_t *buf;
	unsigned *res;
	double *in;
	size_t gotsamples;
	complex *out;
	fftw_plan plan;
};

/* Supported visualizations:
 * 1 -- spectrum
 * 2 -- fountain
 * 3 -- wave
 * 4 -- boom
 * 5 -- solid
 * 6 -- spectro
 */
static void draw_spectrum(struct frame *fr);
static void draw_fountain(struct frame *fr);
static void draw_wave(struct frame *fr);
static void draw_boom(struct frame *fr);
static void draw_solid(struct frame *fr);
static void draw_spectro(struct frame *fr);
static struct visual {
	void (* draw)(struct frame *fr);
	int dft;    /* needs the DFT */
	int color;  /* supports colors */
	int stereo; /* supports stereo */
} visuals[] = {
	{ draw_spectrum, 1, 1, 0 },
	{ draw_fountain, 1, 1, 0 },
	{ draw_wave,     0, 0, 0 },
	{ draw_boom,     0, 1, 0 },
	{ draw_solid,    0, 1, 1 },
	{ draw_spectro,  1, 0, 0 },
};
static int vidx = 0; /* default visual index */

/* We assume the screen is 100 pixels in the y direction.
 * To follow the curses convention (0, 0) is in the top left
 * corner of the screen.  The `min' and `max' values correspond
 * to percentages.  To illustrate this the [0, 20) range gives
 * the top 20% of the screen to the color red.  These values
 * are scaled automatically in the draw() routine to the actual
 * size of the terminal window. */
static struct color_range {
	short pair; /* index in the color table */
	int min;    /* min % */
	int max;    /* max % */
	short fg;   /* foreground color */
	short bg;   /* background color */

	/* these are calculated internally, do not set */
	int scaled_min;
	int scaled_max;
} color_ranges[] = {
	{ 1, 0,  20,  COLOR_RED,    -1, 0, 0 },
	{ 2, 20, 60,  COLOR_YELLOW, -1, 0, 0 },
	{ 3, 60, 100, COLOR_GREEN,  -1, 0, 0 }
};

static void
clearall(struct frame *fr)
{
	unsigned i;

	fr->gotsamples = 0;

	for (i = 0; i < nsamples; i++)
		fr->in[i] = 0.;
	for (i = 0; i < dftout; i++)
		fr->out[i] = 0. + 0. * I;
}

static void
init(struct frame *fr)
{
	fr->fd = open(fname, O_RDONLY | O_NONBLOCK);
	if (fr->fd == -1)
		err(1, "open %s", fname);

	fr->buf = malloc(nsamples * sizeof(int16_t));
	fr->in = malloc(nsamples * sizeof(double));

	/* these are used only by DFT visuals */
	fr->out = malloc(dftout * sizeof(complex));
	fr->res = malloc(dftout * sizeof(unsigned));

	clearall(fr);

	/* we expect single channel input, so half the samples */
	fr->plan = fftw_plan_dft_r2c_1d(dftlen, fr->in, fr->out,
					FFTW_ESTIMATE);
}

static void
done(struct frame *fr)
{
	fftw_destroy_plan(fr->plan);
	free(fr->out);
	free(fr->in);

	free(fr->res);
	free(fr->buf);
	free(fr->peak);
	free(fr->sav);

	close(fr->fd);
}

static void
update(struct frame *fr)
{
	ssize_t n;
	unsigned i;

	n = read(fr->fd, fr->buf, nsamples * sizeof(int16_t));
	if (n == -1) {
		clearall(fr);
		return;
	}

	fr->gotsamples = n / sizeof(int16_t);
}

static void
stagestereo(struct frame *fr)
{
	unsigned i;

	for (i = 0; i < nsamples; i++)
		fr->in[i] = fr->buf[i];
}

static void
stagemono(struct frame *fr)
{
	unsigned i;

	/* we have half the samples after the merge */
	fr->gotsamples /= 2;

	for (i = 0; i < nsamples; i++) {
		fr->in[i] = 0.;
		if (i < fr->gotsamples) {
			/* average the two channels */
			fr->in[i] = fr->buf[i * 2 + 0];
			fr->in[i] += fr->buf[i * 2 + 1];
			fr->in[i] /= 2.;
		}
	}
}

static void
computedft(struct frame *fr)
{
	fftw_execute(fr->plan);
}

static void
setcolor(int on, int y)
{
	unsigned i;
	struct color_range *cr;

	if (!colors)
		return;

	for (i = 0; i < LEN(color_ranges); i++) {
		cr = &color_ranges[i];
		if (y >= cr->scaled_min && y < cr->scaled_max) {
			if (on)
				attron(COLOR_PAIR(cr->pair));
			else
				attroff(COLOR_PAIR(cr->pair));
			return;
		}
	}
}

static void
draw_spectrum(struct frame *fr)
{
	unsigned i, j;
	unsigned freqs_per_col;
	struct color_range *cr;

	/* read dimensions to catch window resize */
	fr->width = COLS;
	fr->height = LINES;

	if (peaks) {
		/* change in width needs new peaks */
		if (fr->width != fr->width_old) {
			fr->peak = realloc(fr->peak, fr->width * sizeof(int));
			for (i = 0; i < fr->width; i++)
				fr->peak[i] = PK_HIDDEN;
			fr->width_old = fr->width;
		}
	}

	if (colors) {
		/* scale color ranges */
		for (i = 0; i < LEN(color_ranges); i++) {
			cr = &color_ranges[i];
			cr->scaled_min = cr->min * fr->height / 100;
			cr->scaled_max = cr->max * fr->height / 100;
		}
	}

	/* take most of the low part of the band */
	freqs_per_col = dftout / fr->width;
	freqs_per_col *= 0.8;

	/* scale each frequency to screen */
	for (i = 0; i < dftout; i++) {
		/* complex absolute value */
		fr->res[i] = cabs(fr->out[i]);
		/* normalize it */
		fr->res[i] /= dftlen;
		/* boost higher freqs */
		fr->res[i] *= log2(i);
		fr->res[i] *= 0.00005 * i;
		fr->res[i] = pow(fr->res[i], 0.5);
		/* scale it */
		fr->res[i] *= 0.15 * fr->height;
	}

	erase();
	attron(A_BOLD);
	for (i = 0; i < fr->width; i++) {
		size_t bar_height = 0;
		size_t ybegin, yend;

		/* compute bar height */
		for (j = 0; j < freqs_per_col; j++)
			bar_height += fr->res[i * freqs_per_col + j];
		bar_height /= freqs_per_col;

		/* we draw from top to bottom */
		ybegin = fr->height - MIN(bar_height, fr->height);
		yend = fr->height;

		/* If the current freq reaches the peak, the peak is
		 * updated to that height, else it drops by one line. */
		if (peaks) {
			if (fr->peak[i] >= ybegin)
				fr->peak[i] = ybegin;
			else
				fr->peak[i]++;
			/* this freq died out */
			if (fr->height == ybegin && fr->peak[i] == ybegin)
				fr->peak[i] = PK_HIDDEN;
		}

		/* output bars */
		for (j = ybegin; j < yend; j++) {
			move(j, i);
			setcolor(1, j);
			printw("%lc", chbar);
			setcolor(0, j);
		}

		/* output peaks */
		if (peaks && fr->peak[i] != PK_HIDDEN) {
			move(fr->peak[i], i);
			setcolor(1, fr->peak[i]);
			printw("%lc", chpeak);
			setcolor(0, fr->peak[i]);
		}
	}
	attroff(A_BOLD);
	refresh();
}

static void
draw_wave(struct frame *fr)
{
	unsigned i, j;
	unsigned samples_per_col;
	double pt_pos, pt_pos_prev = 0, pt_pos_mid;

	/* read dimensions to catch window resize */
	fr->width = COLS;
	fr->height = LINES;

	erase();

	/* not enough samples */
	if (fr->gotsamples < fr->width)
		return;

	samples_per_col = fr->gotsamples / fr->width;

	attron(A_BOLD);
	for (i = 0; i < fr->width; i++) {
		size_t y;

		/* compute point position */
		pt_pos = 0;
		for (j = 0; j < samples_per_col; j++)
			pt_pos += fr->in[i * samples_per_col + j];
		pt_pos /= samples_per_col;
		/* normalize it */
		pt_pos /= INT16_MAX;
		/* scale it */
		pt_pos *= (fr->height / 2) * 0.8;

		/* output points */
		y = fr->height / 2 + pt_pos; /* centering */
		move(y, i);
		printw("%lc", chpoint);

		/* Output a helper point by averaging with the previous
		 * position.  This creates a nice effect.  We don't care
		 * about overlaps with the current point. */
		pt_pos_mid = (pt_pos_prev + pt_pos) / 2.0;
		y = fr->height / 2 + pt_pos_mid; /* centering */
		move(y, i);
		printw("%lc", chpoint);

		pt_pos_prev = pt_pos;
	}
	attroff(A_BOLD);
	refresh();
}

static void
draw_fountain(struct frame *fr)
{
	unsigned i, j;
	struct color_range *cr;
	static int col = 0;
	size_t bar_height = 0;
	unsigned freqs;

	/* read dimensions to catch window resize */
	fr->width = COLS;
	fr->height = LINES;

	/* change in width needs new keep state */
	if (fr->width != fr->width_old) {
		fr->sav = realloc(fr->sav, fr->width * sizeof(int));
		for (i = 0; i < fr->width; i++)
			fr->sav[i] = fr->height;
		fr->width_old = fr->width;
	}

	if (colors) {
		/* scale color ranges */
		for (i = 0; i < LEN(color_ranges); i++) {
			cr = &color_ranges[i];
			cr->scaled_min = cr->min * fr->height / 100;
			cr->scaled_max = cr->max * fr->height / 100;
		}
	}

	/* scale each frequency to screen */
	for (i = 0; i < dftout; i++) {
		/* complex absolute value */
		fr->res[i] = cabs(fr->out[i]);
		/* normalize it */
		fr->res[i] /= dftlen;
		/* scale it */
		fr->res[i] *= 0.006 * fr->height;
	}

	/* take most of the low part of the band */
	freqs = dftout / fr->width;
	freqs *= 0.8;

	/* compute bar height */
	for (j = 0; j < freqs; j++)
		bar_height += fr->res[j];
	bar_height /= freqs;

	erase();
	attron(A_BOLD);

	/* ensure we are inside the frame */
	col %= fr->width;

	for (i = 0; i < fr->width; i++) {
		size_t ybegin, yend;

		/* we draw from top to bottom */
		if (i == col) {
			ybegin = fr->height - MIN(bar_height, fr->height);
			fr->sav[col] = ybegin;
		} else {
			if (keep)
				ybegin = fr->sav[i];
			else
				ybegin = fr->sav[i]++;
		}
		yend = fr->height;

		/* output bars */
		for (j = ybegin; j < yend; j++) {
			move(j, i);
			setcolor(1, j);
			printw("%lc", chbar);
			setcolor(0, j);
		}
	}

	/* current column bounces back */
	if (bounce)
		if (left)
			if (col == 0)
				left = 0;
			else
				col--;
		else
			if (col == fr->width - 1)
				left = 1;
			else
				col++;
	/* current column wraps around */
	else
		if (left)
			col = (col == 0) ? fr->width - 1 : col - 1;
		else
			col = (col == fr->width - 1) ? 0 : col + 1;

	attroff(A_BOLD);
	refresh();
}

static void
draw_boom(struct frame *fr)
{
	unsigned i, j;
	struct color_range *cr;
	unsigned dim, cx, cy, cur, r;
	double avg = 0;

	erase();

	/* no samples at all */
	if (fr->gotsamples == 0)
		return;

	/* read dimensions to catch window resize */
	fr->width = COLS;
	fr->height = LINES;

	if (colors) {
		/* scale color ranges */
		for (i = 0; i < LEN(color_ranges); i++) {
			cr = &color_ranges[i];
			cr->scaled_min = cr->min * fr->height / 100;
			cr->scaled_max = cr->max * fr->height / 100;
		}
	}

	/* We assume that to draw a circle using a monospace font we need
	 * _twice_ the distance on the x-axis, so we double everything. */

	/* size of radius */
	dim = MIN(fr->width / 2, fr->height);

	for (i = 0; i < fr->gotsamples; i++)
		avg += abs(fr->in[i]);
	avg /= fr->gotsamples;
	/* scale it to our box */
	r = (avg * dim / INT16_MAX);

	/* center */
	cx = fr->width / 2;
	cy = fr->height / 2;

	attron(A_BOLD);
	setcolor(1, fr->height - 3 * r);
	/* put the center point */
	move(cy, cx);
	printw("%lc", chpoint);
	for (i = 0; i < fr->width; i++) {
		for (j = 0; j < fr->height; j++) {
			cur = sqrt((i - cx) * (i - cx) +
			           (j - cy) * (j - cy));
			/* draw points on the perimeter */
			if (cur == r) {
				move(j, 2 * i - cx);
				printw("%lc", chpoint);
				/* leave just the center point alone */
				if (i == cx && j == cy)
					continue;
				/* draw second point to make line thicker */
				if (i <= cx) {
					move(j, 2 * i - cx - 1);
					printw("%lc", chpoint);
				}
				if (i >= cx) {
					move(j, 2 * i - cx + 1);
					printw("%lc", chpoint);
				}
			}
		}
	}
	setcolor(0, fr->height - 3 * r);
	attroff(A_BOLD);
	refresh();
}

static void
draw_solid(struct frame *fr)
{
	unsigned i, j;
	struct color_range *cr;
	unsigned samples_per_col;
	double pt_pos, pt_l, pt_r;

	/* read dimensions to catch window resize */
	fr->width = COLS;
	fr->height = LINES;

	if (colors) {
		/* scale color ranges */
		for (i = 0; i < LEN(color_ranges); i++) {
			cr = &color_ranges[i];
			cr->scaled_min = cr->min * (fr->height / 2) / 100;
			cr->scaled_max = cr->max * (fr->height / 2) / 100;
		}
	}

	erase();

	/* not enough samples */
	if (fr->gotsamples < fr->width)
		return;

	samples_per_col = fr->gotsamples / fr->width;

	attron(A_BOLD);
	for (i = 0; i < fr->width; i++) {
		size_t y;

		/* compute point position */
		if (stereo) {
			/* round down to an even */
			if (samples_per_col % 2 != 0)
				samples_per_col--;
			pt_l = pt_r = 0;
			for (j = 0; j < samples_per_col; j += 2) {
				pt_l += fr->in[i * samples_per_col + j + 0];
				pt_r += fr->in[i * samples_per_col + j + 1];
			}
			pt_l /= samples_per_col / 2;
			pt_r /= samples_per_col / 2;
			/* normalize it */
			pt_l /= INT16_MAX;
			pt_r /= INT16_MAX;
			/* scale it */
			pt_l *= (fr->height / 2);
			pt_r *= (fr->height / 2);
		} else {
			pt_pos = 0;
			for (j = 0; j < samples_per_col; j++)
				pt_pos += fr->in[i * samples_per_col + j];
			pt_pos /= samples_per_col;
			/* normalize it */
			pt_pos /= INT16_MAX;
			/* scale it */
			pt_pos *= (fr->height / 2);
			/* treat left and right as the same */
			pt_l = pt_pos;
			pt_r = pt_pos;
		}

		/* output points */
		setcolor(1, fr->height / 2 - MAX(abs(pt_l), abs(pt_r)));
		for (y = fr->height / 2 - abs(pt_l);
		    y <= fr->height / 2 + abs(pt_r);
		    y++) {
			move(y, i);
			printw("%lc", chbar);
		}
		setcolor(0, fr->height / 2 - MAX(abs(pt_l), abs(pt_r)));
	}
	attroff(A_BOLD);
	refresh();
}

static void
draw_spectro(struct frame *fr)
{
	unsigned i, j;
	unsigned freqs_per_row;
	static int col = 0;

	/* read dimensions to catch window resize */
	fr->width = COLS;
	fr->height = LINES;

	/* reset on window resize */
	if (fr->width != fr->width_old || fr->height != fr->height_old) {
		erase();
		fr->width_old = fr->width;
		fr->height_old = fr->height;
		col = 0;
	}

	/* take most of the low part of the band */
	freqs_per_row = dftout / fr->width;
	freqs_per_row *= 0.8;

	/* normalize each frequency */
	for (i = 0; i < dftout; i++) {
		/* complex absolute value */
		fr->res[i] = cabs(fr->out[i]);
		/* normalize it */
		fr->res[i] /= dftlen;
		/* boost higher freqs */
		fr->res[i] *= log2(i);
		fr->res[i] = pow(fr->res[i], 0.5);
		/* scale it */
		fr->res[i] *= WCSLEN(intensity) * 0.04;
	}

	/* ensure we are inside the frame */
	col %= fr->width;

	attron(A_BOLD);
	for (j = 0; j < fr->height; j++) {
		size_t amplitude = 0;

		/* compute amplitude */
		for (i = 0; i < freqs_per_row; i++)
			amplitude += fr->res[j * freqs_per_row + i];
		amplitude /= freqs_per_row;
		if (amplitude > WCSLEN(intensity))
			amplitude = WCSLEN(intensity) - 1;

		/* output intensity */
		move(fr->height - j - 1, col);
		printw("%lc", intensity[amplitude]);
	}
	attroff(A_BOLD);
	refresh();

	col = (col == fr->width - 1) ? 0 : col + 1;
}

static void
initcolors(void)
{
	unsigned i;
	struct color_range *cr;

	start_color();
	for (i = 0; i < LEN(color_ranges); i++) {
		cr = &color_ranges[i];
		init_pair(cr->pair, cr->fg, cr->bg);
	}
}

static void
usage(void)
{
	fprintf(stderr, "usage: %s [-hcpklbs] [-d num] [fifo]\n", argv0);
	fprintf(stderr, "default fifo path is `/tmp/audio.fifo'\n");
	exit(1);
}

int
main(int argc, char *argv[])
{
	int c;
	struct frame fr;
	int vidx_prev;
	int fd;

	argv0 = argv[0];
	while (--argc > 0 && (*++argv)[0] == '-')
		while ((c = *++argv[0]))
			switch (c) {
			case 'd':
				if (*++argv == NULL)
					usage();
				argc--;
				vidx = *argv[0] - '0' - 1;
				if (vidx < 0 || vidx > LEN(visuals) - 1)
					errx(1, "illegal visual index");
				break;
			case 'c':
				colors = 1;
				break;
			case 'p':
				peaks = 1;
				break;
			case 'k':
				keep = 1;
				break;
			case 'l':
				left = 1;
				break;
			case 'b':
				bounce = 1;
				break;
			case 's':
				stereo = 1;
				break;
			case 'h':
				/* fall-through */
			default:
				usage();
			}
	if (argc == 1)
		fname = argv[0];
	else if (argc > 1)
		usage();

	/* init frame context */
	memset(&fr, 0, sizeof(fr));
	init(&fr);

	setlocale(LC_ALL, "");

	/* init curses */
	initscr();
	cbreak();
	noecho();
	nonl();
	intrflush(stdscr, FALSE);
	keypad(stdscr, TRUE);
	curs_set(FALSE); /* hide cursor */
	timeout(msec);
	use_default_colors();

	if (colors && has_colors() == FALSE) {
		endwin();
		done(&fr);
		errx(1, "your terminal does not support colors");
	}

	vidx_prev = vidx;

	while (!die) {
		switch (getch()) {
		case 'q':
			die = 1;
			break;
		case 'c':
			if (has_colors() == TRUE)
				colors = !colors;
			break;
		case 'p':
			peaks = !peaks;
			break;
		case 'k':
			keep = !keep;
			break;
		case 'l':
			left = !left;
			break;
		case 'b':
			bounce = !bounce;
			break;
		case 's':
			stereo = !stereo;
			break;
		case '1':
			vidx = 0;
			break;
		case '2':
			vidx = 1;
			break;
		case '3':
			vidx = 2;
			break;
		case '4':
			vidx = 3;
			break;
		case '5':
			vidx = 4;
			break;
		case '6':
			vidx = 5;
			break;
		case 'n':
		case KEY_RIGHT:
			vidx = vidx == (LEN(visuals) - 1) ? 0 : vidx + 1;
			break;
		case 'N':
		case KEY_LEFT:
			vidx = vidx == 0 ? LEN(visuals) - 1 : vidx - 1;
			break;
		case ' ':
			freeze = !freeze;
			break;
		}

		/* detect visualization change */
		if (vidx != vidx_prev)
			fr.width_old = 0;

		/* not all visuals support colors */
		if (colors && visuals[vidx].color)
			initcolors();
		else
			(void)use_default_colors();

		update(&fr);

		/* may need to merge channels */
		if (stereo && visuals[vidx].stereo)
			stagestereo(&fr);
		else
			stagemono(&fr);

		/* compute the DFT if needed */
		if (visuals[vidx].dft)
			computedft(&fr);

		if (!freeze)
			visuals[vidx].draw(&fr);

		vidx_prev = vidx;
	}

	endwin(); /* restore terminal */

	done(&fr); /* destroy context */

	return (0);
}
