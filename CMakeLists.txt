cmake_minimum_required (VERSION 3.8)
project ("C_Number_Recognition")

# 컴파일러별 경고 및 최적화 옵션 분리
if(MSVC)
    # 공통 경고 설정
    add_compile_options(/W3 /WX)
    
    # 빌드 타입별로 옵션 구분
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2)
    else()
        add_compile_options(/Od /RTC1)
    endif()
else()
    # GCC/Clang용
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -Wall -Wextra -Werror)
    else()
        add_compile_options(-O0 -g -Wall -Wextra -Werror)
    endif()
endif()

# 실행 파일 1: train
add_executable(
    train
    src/train.c
    "src/DRAW_INPUT.h" "src/DRAW_INPUT.c"
    "src/NN.h" "src/NN.c"
)

# 실행 파일 2: predict
add_executable(
    predict
    src/main.c
    "src/DRAW_INPUT.h" "src/DRAW_INPUT.c"
    "src/NN.h" "src/NN.c"
)