#include "gtcrn.h"

#include <string.h>

void gtcrn_state_reset(gtcrn_state_t *state) {
    if (state != 0) memset(state, 0, sizeof(*state));
}
