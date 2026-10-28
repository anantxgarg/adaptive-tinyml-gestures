/*
 * Host-side device simulator. Speaks a line protocol on stdin/stdout so the Python
 * demo can drive the exact same C code that would run on the MCU.
 *
 *   P <300 floats>          predict window  -> "<class> <confidence>"
 *   E <300 floats>          int8 embedding  -> "<16 ints>"            (parity check)
 *   L <label> <300 floats>  learn on-device -> "OK" | "ERR"
 *   A                       add new class   -> "<id>" | "-1"
 *   R                       reset to factory-> "OK"
 *   I                       info            -> "classes=.. flash=.. ram=.."
 */
#include <stdio.h>
#include <stdlib.h>
#include "tinyml.h"

static char line[16384];

static int parse_window(char *p, float win[TML_WIN][3])
{
    for (int i = 0; i < TML_WIN * 3; i++) {
        char *end;
        win[i / 3][i % 3] = strtof(p, &end);
        if (end == p) return -1;
        p = end;
    }
    return 0;
}

int main(void)
{
    static float win[TML_WIN][3];
    tml_init();

    while (fgets(line, sizeof line, stdin)) {
        char *p = line + 1;
        switch (line[0]) {
        case 'P': {
            float conf;
            if (parse_window(p, win)) { puts("ERR"); break; }
            int c = tml_predict(win, &conf);
            printf("%d %.4f\n", c, conf);
            break;
        }
        case 'E': {
            int8_t emb[TML_EMB_DIM];
            if (parse_window(p, win)) { puts("ERR"); break; }
            tml_embed(win, emb);
            for (int i = 0; i < TML_EMB_DIM; i++) printf(i ? " %d" : "%d", emb[i]);
            putchar('\n');
            break;
        }
        case 'L': {
            int label = (int)strtol(p, &p, 10);
            if (parse_window(p, win)) { puts("ERR"); break; }
            puts(tml_learn(win, label) ? "ERR" : "OK");
            break;
        }
        case 'A': printf("%d\n", tml_add_class()); break;
        case 'R': tml_init(); puts("OK"); break;
        case 'I':
            printf("classes=%d flash=%zu ram=%zu\n", tml_num_classes(),
                   tml_flash_bytes(), tml_ram_bytes());
            break;
        default: puts("ERR");
        }
        fflush(stdout);
    }
    return 0;
}
