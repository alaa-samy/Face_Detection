{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled10.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyOyXV+noqVj09IieEwweC5q",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/alaa-samy/Face_Detection/blob/main/main.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "E9ER09qtqhts",
        "outputId": "d3bd8ef8-3a4a-4e05-f985-3f90e9f2e2b6"
      },
      "source": [
        "pip install opencv-python-headless"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Collecting opencv-python-headless\n",
            "  Downloading opencv_python_headless-4.5.3.56-cp37-cp37m-manylinux2014_x86_64.whl (37.1 MB)\n",
            "\u001b[K     |████████████████████████████████| 37.1 MB 44 kB/s \n",
            "\u001b[?25hRequirement already satisfied: numpy>=1.14.5 in /usr/local/lib/python3.7/dist-packages (from opencv-python-headless) (1.19.5)\n",
            "Installing collected packages: opencv-python-headless\n",
            "Successfully installed opencv-python-headless-4.5.3.56\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 547
        },
        "id": "bPrisQ7dq9n2",
        "outputId": "b8e0375d-7a99-4f07-a752-a778c0a80193"
      },
      "source": [
        "# Install modules\n",
        "# pip install opencv-python-headless\n",
        "\n",
        "\n",
        "# Import modules\n",
        "import cv2\n",
        "from google.colab.patches import cv2_imshow\n",
        "from random import randrange\n",
        "\n",
        "# Load pre-trained data\n",
        "trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')\n",
        "\n",
        "# Image to detect faces\n",
        "img = cv2.imread('people.jpg')\n",
        "\n",
        "# Make grayscale images\n",
        "grayscale_img =cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) \n",
        "\n",
        "# Detect faces\n",
        "face_coordinates = trained_face_data.detectMultiScale(grayscale_img)\n",
        "\n",
        "# Draw rectangle around faces\n",
        "for (x,y,w,h) in face_coordinates:\n",
        "  cv2.rectangle(img , (x,y) , (x+w , y+h) , (randrange(256), randrange(256) , randrange(256)) , 2)\n",
        "\n",
        "cv2_imshow(img)\n",
        "\n",
        "# print(face_coordinates)\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<PIL.Image.Image image mode=RGB size=1920x1080 at 0x7FE0C2043710>"
            ]
          },
          "metadata": {
            "tags": []
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "EbGSc2MtrTGS"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}