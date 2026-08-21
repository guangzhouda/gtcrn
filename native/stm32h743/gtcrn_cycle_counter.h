#ifndef GTCRN_CYCLE_COUNTER_H
#define GTCRN_CYCLE_COUNTER_H

/* Include the STM32/CMSIS device header before this file. */
static inline void gtcrn_cycle_counter_init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static inline uint32_t gtcrn_cycle_counter_now(void) {
    return DWT->CYCCNT;
}

#endif
