#pragma comment(linker, "/SUBSYSTEM:WINDOWS")

#define WIN32_LEAN_AND_MEAN

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <windows.h>

#include "NN.h"
#include "DRAW_INPUT.h"

#define DEBUG

#define ID_BUTTON_CLEAR   1001
#define ID_BUTTON_PREDICT 1002
#define ID_BUTTON_QUIT    1003

NN nn;
uint8_t quit = 0;

static void ShowErrorMessage(const char* message)
{
    MessageBoxA(NULL, message, "Error", MB_ICONERROR | MB_OK);
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    static HDC hMemDC = NULL;
    static HBITMAP hBitmap = NULL;
    static HGDIOBJ oldBitmap = NULL;

    static HWND hwndButtonClear = NULL;
    static HWND hwndButtonPredict = NULL;
    static HWND hwndButtonQuit = NULL;

    static uint8_t* data = NULL;
    static uint8_t drawing = 0;

    switch (msg)
    {
        case WM_CREATE:
        {
            HDC hdc = GetDC(hwnd);
            hMemDC = CreateCompatibleDC(hdc);
            hBitmap = CreateCompatibleBitmap(hdc, CANVAS_SIZE, CANVAS_SIZE);
            oldBitmap = SelectObject(hMemDC, hBitmap);

            data = (uint8_t*) malloc(sizeof(uint8_t) * CELL_LEN * CELL_LEN);
            ClearCanvas(data);

            DrawInCanvas(hMemDC, data);
            ReleaseDC(hwnd, hdc);

            DWORD btnStyle = WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON;

            hwndButtonClear = CreateWindow(
                "BUTTON", "Clear", btnStyle, 
                (int)(CANVAS_X + CANVAS_SIZE * 0.0625f), 
                (int)(CANVAS_Y + CANVAS_SIZE * 1.03125f),
                (int)(CANVAS_SIZE / 4), 
                (int)(CANVAS_SIZE / 8), 
                hwnd, (HMENU)ID_BUTTON_CLEAR, NULL, NULL
            );

            hwndButtonPredict = CreateWindow(
                "BUTTON", "Predict", btnStyle, 
                (int)(CANVAS_X + CANVAS_SIZE * 0.375f), 
                (int)(CANVAS_Y + CANVAS_SIZE * 1.03125f), 
                (int)(CANVAS_SIZE / 4), 
                (int)(CANVAS_SIZE / 8), 
                hwnd, (HMENU)ID_BUTTON_PREDICT, NULL, NULL
            );

            hwndButtonQuit = CreateWindow(
                "BUTTON", "Quit", btnStyle, 
                (int)(CANVAS_X + CANVAS_SIZE * 0.6875f), 
                (int)(CANVAS_Y + CANVAS_SIZE * 1.03125f),
                (int)(CANVAS_SIZE / 4), 
                (int)(CANVAS_SIZE / 8), 
                hwnd, (HMENU)ID_BUTTON_QUIT, NULL, NULL
            );

            return 0;
        }

        case WM_SIZE:
        {
            RECT rect;
            GetClientRect(hwnd, &rect);

            SetScreenConstants(rect.right - rect.left, rect.bottom - rect.top);

            if (hMemDC && hBitmap)
            {
                SelectObject(hMemDC, oldBitmap);
                DeleteObject(hBitmap);
                DeleteDC(hMemDC);

                hMemDC = NULL;
                hBitmap = NULL;
                oldBitmap = NULL;
            }

            HDC hdc = GetDC(hwnd);
            hMemDC = CreateCompatibleDC(hdc);
            hBitmap = CreateCompatibleBitmap(hdc, CANVAS_SIZE, CANVAS_SIZE);
            oldBitmap = SelectObject(hMemDC, hBitmap);

            ReleaseDC(hwnd, hdc);

            DrawInCanvas(hMemDC, data);

            return 0;
        }

        case WM_LBUTTONDOWN:
        {
            drawing = 1;

            return 0;
        }

        case WM_MOUSEMOVE:
        {
            if (drawing && hMemDC != NULL)
            {
                int x = LOWORD(lParam);
                int y = HIWORD(lParam);

                DrawCircleInCanvas(data, x, y);
                DrawInCanvas(hMemDC, data);

                RECT crect = { CANVAS_X, CANVAS_Y, CANVAS_X + CANVAS_SIZE, CANVAS_Y + CANVAS_SIZE };
                InvalidateRect(hwnd, &crect, FALSE);
            }

            return 0;
        }

        case WM_LBUTTONUP:
        {
            drawing = 0;

            return 0;
        }

        case WM_COMMAND:
        {
            switch (LOWORD(wParam))
            {
                case ID_BUTTON_CLEAR:
                {
                    if (hMemDC && data)
                    {
                        ClearCanvas(data);
                        DrawInCanvas(hMemDC, data);

                        InvalidateRect(hwnd, NULL, TRUE);
                    }

                    break;
                }

                case ID_BUTTON_PREDICT:
                {
                    /* if (hMemDC != NULL && hBitmap != NULL)
                    {
                        if (input)
                        {
                            for (int i = 0; i < CELL_LEN * CELL_LEN; i++)
                                MAT_AT(NN_INPUT(nn), 0, i) = input[i];

                            free(input);

                            NN_forward(nn);
                            printf("Prediction:\n");

                            for (int i = 0; i < 10; i++)
                                printf("Digit %d: %.3f\n", i, MAT_AT(NN_OUTPUT(nn), 0, i));

                            printf("-----------------\n");
                        }
                    } */

                    break;
                }

                case ID_BUTTON_QUIT:
                {
                    quit = 1;

                    PostQuitMessage(0);
                    
                    break;
                }
            }

            return 0;
        }

        case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);

            if (hMemDC != NULL)
                BitBlt(hdc, CANVAS_X, CANVAS_Y, CANVAS_SIZE, CANVAS_SIZE, hMemDC, 0, 0, SRCCOPY);

            EndPaint(hwnd, &ps);

            return 0;
        }

        case WM_DESTROY:
        {
            if (hMemDC != NULL)
            {
                SelectObject(hMemDC, oldBitmap);
                DeleteObject(hBitmap);
                DeleteDC(hMemDC);

                hMemDC = NULL;
                hBitmap = NULL;
                oldBitmap = NULL;
            }

            if (data != NULL)
            {
                free(data);

                data = NULL;
            }

            PostQuitMessage(0);

            return 0;
        }
    }

    return DefWindowProc(hwnd, msg, wParam, lParam);
}

int WINAPI WinMain(
    _In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPSTR lpCmdLine,
    _In_ int nCmdShow
)
{
    (void)hPrevInstance;
    (void)lpCmdLine;

    // 모델 파일 불러오기
    /*FILE* file = fopen("data/model.nn", "rb");
    if (!file)
    {
        MessageBox(NULL, "Failed to open model.nn", "Error", MB_ICONERROR);
        return EXIT_FAILURE;
    }
    nn = NN_load(file);
    fclose(file);

    if (nn.count == 0)
    {
        MessageBox(NULL, "Failed to load neural network model", "Error", MB_ICONERROR);
        return EXIT_FAILURE;
    }*/

    printf("Draw digit with mouse.\nPress ENTER to recognize, C to clear, ESC to quit.\n");

    // 윈도우 등록 및 생성
    WNDCLASS wc = { 0 };
    
    wc.style = 0;
    wc.lpfnWndProc = (WNDPROC) WndProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = hInstance;

    wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    
    wc.hbrBackground = CreateSolidBrush(RGB(32, 32, 32));
    
    wc.lpszMenuName = NULL;
    wc.lpszClassName = "Number_Recognition";

    if (!RegisterClass(&wc))
        return EXIT_FAILURE;

    SetScreenConstants(GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));

#ifdef DEBUG
    HWND hWnd = CreateWindow(
        "Number_Recognition", "Number Recognition",
        WS_SIZEBOX,
        0, 0, SCREEN_WIDTH, SCREEN_HEIGHT,
        NULL, NULL, hInstance, NULL
    );
#else
    HWND hWnd = CreateWindow(
        "Number_Recognition", "Number Recognition",
        WS_POPUP,
        0, 0, SCREEN_WIDTH, SCREEN_HEIGHT,
        NULL, NULL, hInstance, NULL
    );
#endif

    if (!hWnd)
    {
        ShowErrorMessage("Failed to create window");

        return EXIT_FAILURE;
    }

#ifndef DEBUG
    FreeConsole();
#endif

    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);

    MSG msg;
    while (!quit && GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    //NN_free(nn);

    return EXIT_SUCCESS;
}