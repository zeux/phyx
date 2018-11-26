#define MICROPROFILE_HELP_ALT "Right-Click"
#define MICROPROFILE_HELP_MOD "Ctrl"

#ifdef _WIN32
#include "../glad/glad.h"
#endif

#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>

#ifdef __APPLE__
#define GL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED
#include <OpenGL/gl3.h>
#endif

#define MICROPROFILE_WEBSERVER 1
#define MICROPROFILE_GPU_TIMERS_GL 1

#define MICROPROFILE_IMPL
#include "microprofile.h"

#define MICROPROFILEUI_IMPL
#include "microprofileui.h"

#define MICROPROFILEDRAW_IMPL
#include "microprofiledraw.h"

#ifdef __linux__
#define GL_PROC(ret, name, args, argcall) \
	ret name args { \
		static ret (*ptr) args = reinterpret_cast<ret (*) args>(glfwGetProcAddress(#name)); \
		return ptr argcall; \
	}

GL_PROC(void, glQueryCounter, (GLuint id, GLenum target), (id, target))
GL_PROC(void, glGetQueryObjectui64v, (GLuint id, GLenum pname, GLuint64 *params), (id, pname, params))
#endif
