# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.24.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.24.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build

# Include any dependencies generated for this target.
include CMakeFiles/kompute_array_mult.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/kompute_array_mult.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/kompute_array_mult.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/kompute_array_mult.dir/flags.make

CMakeFiles/kompute_array_mult.dir/src/Main.cpp.o: CMakeFiles/kompute_array_mult.dir/flags.make
CMakeFiles/kompute_array_mult.dir/src/Main.cpp.o: /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/src/Main.cpp
CMakeFiles/kompute_array_mult.dir/src/Main.cpp.o: CMakeFiles/kompute_array_mult.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/kompute_array_mult.dir/src/Main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/kompute_array_mult.dir/src/Main.cpp.o -MF CMakeFiles/kompute_array_mult.dir/src/Main.cpp.o.d -o CMakeFiles/kompute_array_mult.dir/src/Main.cpp.o -c /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/src/Main.cpp

CMakeFiles/kompute_array_mult.dir/src/Main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kompute_array_mult.dir/src/Main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/src/Main.cpp > CMakeFiles/kompute_array_mult.dir/src/Main.cpp.i

CMakeFiles/kompute_array_mult.dir/src/Main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kompute_array_mult.dir/src/Main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/src/Main.cpp -o CMakeFiles/kompute_array_mult.dir/src/Main.cpp.s

# Object files for target kompute_array_mult
kompute_array_mult_OBJECTS = \
"CMakeFiles/kompute_array_mult.dir/src/Main.cpp.o"

# External object files for target kompute_array_mult
kompute_array_mult_EXTERNAL_OBJECTS =

kompute_array_mult: CMakeFiles/kompute_array_mult.dir/src/Main.cpp.o
kompute_array_mult: CMakeFiles/kompute_array_mult.dir/build.make
kompute_array_mult: kompute/src/libkompute.a
kompute_array_mult: /usr/local/lib/libvulkan.dylib
kompute_array_mult: kompute/src/kompute_spdlog/libspdlog.a
kompute_array_mult: kompute/src/kompute_fmt/libfmt.a
kompute_array_mult: CMakeFiles/kompute_array_mult.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable kompute_array_mult"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kompute_array_mult.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/kompute_array_mult.dir/build: kompute_array_mult
.PHONY : CMakeFiles/kompute_array_mult.dir/build

CMakeFiles/kompute_array_mult.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/kompute_array_mult.dir/cmake_clean.cmake
.PHONY : CMakeFiles/kompute_array_mult.dir/clean

CMakeFiles/kompute_array_mult.dir/depend:
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/CMakeFiles/kompute_array_mult.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/kompute_array_mult.dir/depend
