
#ifndef LIBKRIGING_EXPORT_H
#define LIBKRIGING_EXPORT_H

#ifdef LIBKRIGING_STATIC_DEFINE
#  define LIBKRIGING_EXPORT
#  define LIBKRIGING_NO_EXPORT
#else
#  ifndef LIBKRIGING_EXPORT
#    ifdef Kriging_EXPORTS
        /* We are building this library */
#      define LIBKRIGING_EXPORT 
#    else
        /* We are using this library */
#      define LIBKRIGING_EXPORT 
#    endif
#  endif

#  ifndef LIBKRIGING_NO_EXPORT
#    define LIBKRIGING_NO_EXPORT 
#  endif
#endif

#ifndef LIBKRIGING_DEPRECATED
#  define LIBKRIGING_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef LIBKRIGING_DEPRECATED_EXPORT
#  define LIBKRIGING_DEPRECATED_EXPORT LIBKRIGING_EXPORT LIBKRIGING_DEPRECATED
#endif

#ifndef LIBKRIGING_DEPRECATED_NO_EXPORT
#  define LIBKRIGING_DEPRECATED_NO_EXPORT LIBKRIGING_NO_EXPORT LIBKRIGING_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef LIBKRIGING_NO_DEPRECATED
#    define LIBKRIGING_NO_DEPRECATED
#  endif
#endif

#endif /* LIBKRIGING_EXPORT_H */
