/* Temporary diagnostic for #429: raw CPUID / XCR0 bits that OpenBLAS 0.3.20
 * DYNAMIC_ARCH (driver/others/dynamic.c) looks at to pick its kernels. */
#include <cpuid.h>
#include <stdio.h>
#include <string.h>

static unsigned long long xgetbv0(void) {
  unsigned int a, d;
  __asm__ volatile("xgetbv" : "=a"(a), "=d"(d) : "c"(0));
  return ((unsigned long long)d << 32) | a;
}

int main(void) {
  unsigned int a, b, c, d;
  char vendor[13] = {0}, brand[49] = {0};
  __cpuid(0, a, b, c, d);
  unsigned int maxleaf = a;
  memcpy(vendor, &b, 4), memcpy(vendor + 4, &d, 4), memcpy(vendor + 8, &c, 4);
  for (unsigned int i = 0; i < 3; i++) {
    __cpuid(0x80000002 + i, a, b, c, d);
    memcpy(brand + 16 * i, &a, 4), memcpy(brand + 16 * i + 4, &b, 4), memcpy(brand + 16 * i + 8, &c, 4),
        memcpy(brand + 16 * i + 12, &d, 4);
  }
  __cpuid(1, a, b, c, d);
  unsigned int family = (a >> 8) & 0xf, model = (a >> 4) & 0xf, exfamily = (a >> 20) & 0xff, exmodel = (a >> 16) & 0xf,
               stepping = a & 0xf;
  int osxsave = (c >> 27) & 1, avx = (c >> 28) & 1, fma = (c >> 12) & 1, hypervisor = (c >> 31) & 1;
  printf("vendor=%s brand=\"%s\"\n", vendor, brand);
  printf("family=%u model=%u exfamily=%u exmodel=%u stepping=%u (display family=0x%x model=0x%x) hypervisor=%d\n",
         family,
         model,
         exfamily,
         exmodel,
         stepping,
         family == 0xf ? family + exfamily : family,
         (family == 6 || family == 0xf) ? (exmodel << 4) | model : model,
         hypervisor);
  unsigned long long xcr0 = osxsave ? xgetbv0() : 0;
  printf("osxsave=%d xcr0=0x%llx (sse/avx state 0x6: %s, avx512 opmask/zmm state 0xe0: %s, amx tile 0x60000: %s)\n",
         osxsave,
         xcr0,
         (xcr0 & 6) == 6 ? "on" : "off",
         (xcr0 & 0xe0) == 0xe0 ? "on" : "off",
         (xcr0 & 0x60000) == 0x60000 ? "on" : "off");
  if (maxleaf >= 7) {
    __cpuid_count(7, 0, a, b, c, d);
    printf(
        "leaf7.0: avx=%d fma=%d avx2=%d avx512f=%d avx512dq=%d avx512cd=%d avx512bw=%d avx512vl=%d avx512vnni=%d "
        "avx512_fp16=%d amx_bf16=%d amx_tile=%d amx_int8=%d\n",
        avx,
        fma,
        (b >> 5) & 1,
        (b >> 16) & 1,
        (b >> 17) & 1,
        (b >> 28) & 1,
        (b >> 30) & 1,
        (b >> 31) & 1,
        (c >> 11) & 1,
        (d >> 23) & 1,
        (d >> 22) & 1,
        (d >> 24) & 1,
        (d >> 25) & 1);
    __cpuid_count(7, 1, a, b, c, d);
    printf("leaf7.1: avx_vnni=%d avx512_bf16=%d\n", (a >> 4) & 1, (a >> 5) & 1);
  }
  return 0;
}
