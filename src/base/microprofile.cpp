#define MICROPROFILE_HELP_ALT "Right-Click"
#define MICROPROFILE_HELP_MOD "Ctrl"

#include <GLFW/glfw3.h>

#ifdef __APPLE__
#define GL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED
#include <OpenGL/gl3.h>
#endif

#define MICROPROFILE_GPU_TIMERS 0

#define MICROPROFILE_IMPL
#include "microprofile.h"

#define MICROPROFILEUI_IMPL
#include "microprofileui.h"

#define MICROPROFILEDRAW_IMPL
#include "microprofiledraw.h"