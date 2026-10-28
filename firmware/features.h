#ifndef FEATURES_H
#define FEATURES_H

#define TML_WIN 100     /* 2 s @ 50 Hz */
#define TML_N_FEAT 23

void tml_features(const float win[TML_WIN][3], float out[TML_N_FEAT]);

#endif
