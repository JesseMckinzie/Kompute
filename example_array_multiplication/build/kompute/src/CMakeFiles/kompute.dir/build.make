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

# Produce verbose output by default.
VERBOSE = 1

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
include kompute/src/CMakeFiles/kompute.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include kompute/src/CMakeFiles/kompute.dir/compiler_depend.make

# Include the progress variables for this target.
include kompute/src/CMakeFiles/kompute.dir/progress.make

# Include the compile flags for this target's objects.
include kompute/src/CMakeFiles/kompute.dir/flags.make

kompute/src/CMakeFiles/kompute.dir/Algorithm.cpp.o: kompute/src/CMakeFiles/kompute.dir/flags.make
kompute/src/CMakeFiles/kompute.dir/Algorithm.cpp.o: /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/Algorithm.cpp
kompute/src/CMakeFiles/kompute.dir/Algorithm.cpp.o: kompute/src/CMakeFiles/kompute.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object kompute/src/CMakeFiles/kompute.dir/Algorithm.cpp.o"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT kompute/src/CMakeFiles/kompute.dir/Algorithm.cpp.o -MF CMakeFiles/kompute.dir/Algorithm.cpp.o.d -o CMakeFiles/kompute.dir/Algorithm.cpp.o -c /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/Algorithm.cpp

kompute/src/CMakeFiles/kompute.dir/Algorithm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kompute.dir/Algorithm.cpp.i"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/Algorithm.cpp > CMakeFiles/kompute.dir/Algorithm.cpp.i

kompute/src/CMakeFiles/kompute.dir/Algorithm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kompute.dir/Algorithm.cpp.s"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/Algorithm.cpp -o CMakeFiles/kompute.dir/Algorithm.cpp.s

kompute/src/CMakeFiles/kompute.dir/Manager.cpp.o: kompute/src/CMakeFiles/kompute.dir/flags.make
kompute/src/CMakeFiles/kompute.dir/Manager.cpp.o: /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/Manager.cpp
kompute/src/CMakeFiles/kompute.dir/Manager.cpp.o: kompute/src/CMakeFiles/kompute.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object kompute/src/CMakeFiles/kompute.dir/Manager.cpp.o"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT kompute/src/CMakeFiles/kompute.dir/Manager.cpp.o -MF CMakeFiles/kompute.dir/Manager.cpp.o.d -o CMakeFiles/kompute.dir/Manager.cpp.o -c /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/Manager.cpp

kompute/src/CMakeFiles/kompute.dir/Manager.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kompute.dir/Manager.cpp.i"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/Manager.cpp > CMakeFiles/kompute.dir/Manager.cpp.i

kompute/src/CMakeFiles/kompute.dir/Manager.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kompute.dir/Manager.cpp.s"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/Manager.cpp -o CMakeFiles/kompute.dir/Manager.cpp.s

kompute/src/CMakeFiles/kompute.dir/OpAlgoDispatch.cpp.o: kompute/src/CMakeFiles/kompute.dir/flags.make
kompute/src/CMakeFiles/kompute.dir/OpAlgoDispatch.cpp.o: /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpAlgoDispatch.cpp
kompute/src/CMakeFiles/kompute.dir/OpAlgoDispatch.cpp.o: kompute/src/CMakeFiles/kompute.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object kompute/src/CMakeFiles/kompute.dir/OpAlgoDispatch.cpp.o"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT kompute/src/CMakeFiles/kompute.dir/OpAlgoDispatch.cpp.o -MF CMakeFiles/kompute.dir/OpAlgoDispatch.cpp.o.d -o CMakeFiles/kompute.dir/OpAlgoDispatch.cpp.o -c /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpAlgoDispatch.cpp

kompute/src/CMakeFiles/kompute.dir/OpAlgoDispatch.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kompute.dir/OpAlgoDispatch.cpp.i"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpAlgoDispatch.cpp > CMakeFiles/kompute.dir/OpAlgoDispatch.cpp.i

kompute/src/CMakeFiles/kompute.dir/OpAlgoDispatch.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kompute.dir/OpAlgoDispatch.cpp.s"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpAlgoDispatch.cpp -o CMakeFiles/kompute.dir/OpAlgoDispatch.cpp.s

kompute/src/CMakeFiles/kompute.dir/OpMemoryBarrier.cpp.o: kompute/src/CMakeFiles/kompute.dir/flags.make
kompute/src/CMakeFiles/kompute.dir/OpMemoryBarrier.cpp.o: /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpMemoryBarrier.cpp
kompute/src/CMakeFiles/kompute.dir/OpMemoryBarrier.cpp.o: kompute/src/CMakeFiles/kompute.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object kompute/src/CMakeFiles/kompute.dir/OpMemoryBarrier.cpp.o"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT kompute/src/CMakeFiles/kompute.dir/OpMemoryBarrier.cpp.o -MF CMakeFiles/kompute.dir/OpMemoryBarrier.cpp.o.d -o CMakeFiles/kompute.dir/OpMemoryBarrier.cpp.o -c /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpMemoryBarrier.cpp

kompute/src/CMakeFiles/kompute.dir/OpMemoryBarrier.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kompute.dir/OpMemoryBarrier.cpp.i"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpMemoryBarrier.cpp > CMakeFiles/kompute.dir/OpMemoryBarrier.cpp.i

kompute/src/CMakeFiles/kompute.dir/OpMemoryBarrier.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kompute.dir/OpMemoryBarrier.cpp.s"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpMemoryBarrier.cpp -o CMakeFiles/kompute.dir/OpMemoryBarrier.cpp.s

kompute/src/CMakeFiles/kompute.dir/OpTensorCopy.cpp.o: kompute/src/CMakeFiles/kompute.dir/flags.make
kompute/src/CMakeFiles/kompute.dir/OpTensorCopy.cpp.o: /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpTensorCopy.cpp
kompute/src/CMakeFiles/kompute.dir/OpTensorCopy.cpp.o: kompute/src/CMakeFiles/kompute.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object kompute/src/CMakeFiles/kompute.dir/OpTensorCopy.cpp.o"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT kompute/src/CMakeFiles/kompute.dir/OpTensorCopy.cpp.o -MF CMakeFiles/kompute.dir/OpTensorCopy.cpp.o.d -o CMakeFiles/kompute.dir/OpTensorCopy.cpp.o -c /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpTensorCopy.cpp

kompute/src/CMakeFiles/kompute.dir/OpTensorCopy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kompute.dir/OpTensorCopy.cpp.i"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpTensorCopy.cpp > CMakeFiles/kompute.dir/OpTensorCopy.cpp.i

kompute/src/CMakeFiles/kompute.dir/OpTensorCopy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kompute.dir/OpTensorCopy.cpp.s"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpTensorCopy.cpp -o CMakeFiles/kompute.dir/OpTensorCopy.cpp.s

kompute/src/CMakeFiles/kompute.dir/OpTensorSyncDevice.cpp.o: kompute/src/CMakeFiles/kompute.dir/flags.make
kompute/src/CMakeFiles/kompute.dir/OpTensorSyncDevice.cpp.o: /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpTensorSyncDevice.cpp
kompute/src/CMakeFiles/kompute.dir/OpTensorSyncDevice.cpp.o: kompute/src/CMakeFiles/kompute.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object kompute/src/CMakeFiles/kompute.dir/OpTensorSyncDevice.cpp.o"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT kompute/src/CMakeFiles/kompute.dir/OpTensorSyncDevice.cpp.o -MF CMakeFiles/kompute.dir/OpTensorSyncDevice.cpp.o.d -o CMakeFiles/kompute.dir/OpTensorSyncDevice.cpp.o -c /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpTensorSyncDevice.cpp

kompute/src/CMakeFiles/kompute.dir/OpTensorSyncDevice.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kompute.dir/OpTensorSyncDevice.cpp.i"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpTensorSyncDevice.cpp > CMakeFiles/kompute.dir/OpTensorSyncDevice.cpp.i

kompute/src/CMakeFiles/kompute.dir/OpTensorSyncDevice.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kompute.dir/OpTensorSyncDevice.cpp.s"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpTensorSyncDevice.cpp -o CMakeFiles/kompute.dir/OpTensorSyncDevice.cpp.s

kompute/src/CMakeFiles/kompute.dir/OpTensorSyncLocal.cpp.o: kompute/src/CMakeFiles/kompute.dir/flags.make
kompute/src/CMakeFiles/kompute.dir/OpTensorSyncLocal.cpp.o: /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpTensorSyncLocal.cpp
kompute/src/CMakeFiles/kompute.dir/OpTensorSyncLocal.cpp.o: kompute/src/CMakeFiles/kompute.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object kompute/src/CMakeFiles/kompute.dir/OpTensorSyncLocal.cpp.o"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT kompute/src/CMakeFiles/kompute.dir/OpTensorSyncLocal.cpp.o -MF CMakeFiles/kompute.dir/OpTensorSyncLocal.cpp.o.d -o CMakeFiles/kompute.dir/OpTensorSyncLocal.cpp.o -c /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpTensorSyncLocal.cpp

kompute/src/CMakeFiles/kompute.dir/OpTensorSyncLocal.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kompute.dir/OpTensorSyncLocal.cpp.i"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpTensorSyncLocal.cpp > CMakeFiles/kompute.dir/OpTensorSyncLocal.cpp.i

kompute/src/CMakeFiles/kompute.dir/OpTensorSyncLocal.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kompute.dir/OpTensorSyncLocal.cpp.s"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/OpTensorSyncLocal.cpp -o CMakeFiles/kompute.dir/OpTensorSyncLocal.cpp.s

kompute/src/CMakeFiles/kompute.dir/Sequence.cpp.o: kompute/src/CMakeFiles/kompute.dir/flags.make
kompute/src/CMakeFiles/kompute.dir/Sequence.cpp.o: /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/Sequence.cpp
kompute/src/CMakeFiles/kompute.dir/Sequence.cpp.o: kompute/src/CMakeFiles/kompute.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object kompute/src/CMakeFiles/kompute.dir/Sequence.cpp.o"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT kompute/src/CMakeFiles/kompute.dir/Sequence.cpp.o -MF CMakeFiles/kompute.dir/Sequence.cpp.o.d -o CMakeFiles/kompute.dir/Sequence.cpp.o -c /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/Sequence.cpp

kompute/src/CMakeFiles/kompute.dir/Sequence.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kompute.dir/Sequence.cpp.i"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/Sequence.cpp > CMakeFiles/kompute.dir/Sequence.cpp.i

kompute/src/CMakeFiles/kompute.dir/Sequence.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kompute.dir/Sequence.cpp.s"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/Sequence.cpp -o CMakeFiles/kompute.dir/Sequence.cpp.s

kompute/src/CMakeFiles/kompute.dir/Tensor.cpp.o: kompute/src/CMakeFiles/kompute.dir/flags.make
kompute/src/CMakeFiles/kompute.dir/Tensor.cpp.o: /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/Tensor.cpp
kompute/src/CMakeFiles/kompute.dir/Tensor.cpp.o: kompute/src/CMakeFiles/kompute.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object kompute/src/CMakeFiles/kompute.dir/Tensor.cpp.o"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT kompute/src/CMakeFiles/kompute.dir/Tensor.cpp.o -MF CMakeFiles/kompute.dir/Tensor.cpp.o.d -o CMakeFiles/kompute.dir/Tensor.cpp.o -c /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/Tensor.cpp

kompute/src/CMakeFiles/kompute.dir/Tensor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kompute.dir/Tensor.cpp.i"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/Tensor.cpp > CMakeFiles/kompute.dir/Tensor.cpp.i

kompute/src/CMakeFiles/kompute.dir/Tensor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kompute.dir/Tensor.cpp.s"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src/Tensor.cpp -o CMakeFiles/kompute.dir/Tensor.cpp.s

# Object files for target kompute
kompute_OBJECTS = \
"CMakeFiles/kompute.dir/Algorithm.cpp.o" \
"CMakeFiles/kompute.dir/Manager.cpp.o" \
"CMakeFiles/kompute.dir/OpAlgoDispatch.cpp.o" \
"CMakeFiles/kompute.dir/OpMemoryBarrier.cpp.o" \
"CMakeFiles/kompute.dir/OpTensorCopy.cpp.o" \
"CMakeFiles/kompute.dir/OpTensorSyncDevice.cpp.o" \
"CMakeFiles/kompute.dir/OpTensorSyncLocal.cpp.o" \
"CMakeFiles/kompute.dir/Sequence.cpp.o" \
"CMakeFiles/kompute.dir/Tensor.cpp.o"

# External object files for target kompute
kompute_EXTERNAL_OBJECTS =

kompute/src/libkompute.a: kompute/src/CMakeFiles/kompute.dir/Algorithm.cpp.o
kompute/src/libkompute.a: kompute/src/CMakeFiles/kompute.dir/Manager.cpp.o
kompute/src/libkompute.a: kompute/src/CMakeFiles/kompute.dir/OpAlgoDispatch.cpp.o
kompute/src/libkompute.a: kompute/src/CMakeFiles/kompute.dir/OpMemoryBarrier.cpp.o
kompute/src/libkompute.a: kompute/src/CMakeFiles/kompute.dir/OpTensorCopy.cpp.o
kompute/src/libkompute.a: kompute/src/CMakeFiles/kompute.dir/OpTensorSyncDevice.cpp.o
kompute/src/libkompute.a: kompute/src/CMakeFiles/kompute.dir/OpTensorSyncLocal.cpp.o
kompute/src/libkompute.a: kompute/src/CMakeFiles/kompute.dir/Sequence.cpp.o
kompute/src/libkompute.a: kompute/src/CMakeFiles/kompute.dir/Tensor.cpp.o
kompute/src/libkompute.a: kompute/src/CMakeFiles/kompute.dir/build.make
kompute/src/libkompute.a: kompute/src/CMakeFiles/kompute.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX static library libkompute.a"
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && $(CMAKE_COMMAND) -P CMakeFiles/kompute.dir/cmake_clean_target.cmake
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kompute.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
kompute/src/CMakeFiles/kompute.dir/build: kompute/src/libkompute.a
.PHONY : kompute/src/CMakeFiles/kompute.dir/build

kompute/src/CMakeFiles/kompute.dir/clean:
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src && $(CMAKE_COMMAND) -P CMakeFiles/kompute.dir/cmake_clean.cmake
.PHONY : kompute/src/CMakeFiles/kompute.dir/clean

kompute/src/CMakeFiles/kompute.dir/depend:
	cd /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication /Users/jessemckinzie/Documents/GitHub/Kompute/kompute/src /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src /Users/jessemckinzie/Documents/GitHub/Kompute/example_array_multiplication/build/kompute/src/CMakeFiles/kompute.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : kompute/src/CMakeFiles/kompute.dir/depend

