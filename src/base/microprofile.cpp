#define MICROPROFILE_HELP_ALT "Right-Click"
#define MICROPROFILE_HELP_MOD "Ctrl"

#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>

#ifdef __APPLE__
#define GL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED
#include <OpenGL/gl3.h>
#else
#endif

#define MICROPROFILE_WEBSERVER 1
#define MICROPROFILE_GPU_TIMERS_GL 1

#define MICROPROFILE_IMPL
#include "microprofile.h"

#define MICROPROFILEUI_IMPL
#include "microprofileui.h"

#define MICROPROFILEDRAW_IMPL
#include "microprofiledraw.h"
