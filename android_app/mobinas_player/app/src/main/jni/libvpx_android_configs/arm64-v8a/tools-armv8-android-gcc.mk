## Copyright (c) 2011 The WebM project authors. All Rights Reserved.
## 
## Use of this source code is governed by a BSD-style license
## that can be found in the LICENSE file in the root of the source
## tree. An additional intellectual property rights grant can be found
## in the file PATENTS.  All contributing project authors may
## be found in the AUTHORS file in the root of the source tree.
# This file automatically generated by configure. Do not edit!
SRC_PATH="../../libvpx"
SRC_PATH_BARE=../../libvpx
BUILD_PFX=
TOOLCHAIN=armv8-android-gcc
ASM_CONVERSION=../../libvpx/build/make/ads2gas.pl
GEN_VCPROJ=
MSVS_ARCH_DIR=

CC=/home/martin/Desktop/android-ndk-r14b/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin/arm-linux-androideabi-gcc
CXX=/home/martin/Desktop/android-ndk-r14b/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin/arm-linux-androideabi-g++
AR=/home/martin/Desktop/android-ndk-r14b/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin/arm-linux-androideabi-ar
LD=/home/martin/Desktop/android-ndk-r14b/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin/arm-linux-androideabi-g++
AS=/home/martin/Desktop/android-ndk-r14b/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin/arm-linux-androideabi-as
STRIP=/home/martin/Desktop/android-ndk-r14b/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin/arm-linux-androideabi-strip
NM=/home/martin/Desktop/android-ndk-r14b/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin/arm-linux-androideabi-nm

CFLAGS  =  --sysroot=/home/martin/Desktop/android-ndk-r14b/platforms/android-24/arch-arm -DNDEBUG -O3 -fPIC -Wall -Wdeclaration-after-statement -Wdisabled-optimization -Wfloat-conversion -Wpointer-arith -Wtype-limits -Wcast-qual -Wvla -Wimplicit-function-declaration -Wuninitialized -Wunused -Wextra -Wundef -isystem /home/martin/Desktop/android-ndk-r14b/sysroot/usr/include/arm-linux-androideabi -isystem /home/martin/Desktop/android-ndk-r14b/sysroot/usr/include
CXXFLAGS  =  --sysroot=/home/martin/Desktop/android-ndk-r14b/platforms/android-24/arch-arm -DNDEBUG -O3 -fPIC -Wall -Wdisabled-optimization -Wfloat-conversion -Wpointer-arith -Wtype-limits -Wcast-qual -Wvla -Wuninitialized -Wunused -Wextra -isystem /home/martin/Desktop/android-ndk-r14b/sysroot/usr/include/arm-linux-androideabi -isystem /home/martin/Desktop/android-ndk-r14b/sysroot/usr/include
ARFLAGS = -crs$(if $(quiet),,v)
LDFLAGS =  --sysroot=/home/martin/Desktop/android-ndk-r14b/platforms/android-24/arch-arm -Wl,--fix-cortex-a8
ASFLAGS =  --defsym ARCHITECTURE=8 -march=armv8-a
extralibs = 
AS_SFX    = .S
EXE_SFX   = 
VCPROJ_SFX = 
RTCD_OPTIONS = 
fmt_deps = sed -e 's;^\([a-zA-Z0-9_]*\)\.o;${@:.d=.o} $@;'
ARCH_ARM=yes
HAVE_NEON=yes
HAVE_VPX_PORTS=yes
HAVE_PTHREAD_H=yes
HAVE_UNISTD_H=yes
CONFIG_DEPENDENCY_TRACKING=yes
CONFIG_INSTALL_BINS=yes
CONFIG_INSTALL_LIBS=yes
CONFIG_GCC=yes
CONFIG_PIC=yes
CONFIG_MULTITHREAD=yes
CONFIG_VP9_ENCODER=yes
CONFIG_VP9_DECODER=yes
CONFIG_VP9=yes
CONFIG_ENCODERS=yes
CONFIG_DECODERS=yes
CONFIG_SPATIAL_RESAMPLING=yes
CONFIG_REALTIME_ONLY=yes
CONFIG_STATIC=yes
CONFIG_OS_SUPPORT=yes
CONFIG_WEBM_IO=yes
CONFIG_LIBYUV=yes
CONFIG_TEMPORAL_DENOISING=yes
CONFIG_SNPE=yes
HAVE_GNU_STRIP=yes
