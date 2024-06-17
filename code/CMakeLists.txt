cmake_minimum_required(VERSION 3.0)
PROJECT(cec17)
ADD_EXECUTABLE(test "test.cc")
ADD_EXECUTABLE(testrandom "testrandom.cc")
ADD_EXECUTABLE(testsolis "testsolis.cc")

ADD_EXECUTABLE(queen_bee "qbe.cc")
ADD_EXECUTABLE(queen_bee_AM "qbe_am.cc")
ADD_EXECUTABLE(queen_bee_Mejora "qbe_mejora.cc")

ADD_LIBRARY("cec17_test_func" SHARED "cec17_test_func.c" "cec17.c")
TARGET_LINK_LIBRARIES(test "cec17_test_func")
TARGET_LINK_LIBRARIES(testrandom "cec17_test_func")
TARGET_LINK_LIBRARIES(testsolis "cec17_test_func")

TARGET_LINK_LIBRARIES(queen_bee "cec17_test_func")
TARGET_LINK_LIBRARIES(queen_bee_AM "cec17_test_func")
TARGET_LINK_LIBRARIES(queen_bee_Mejora "cec17_test_func")
