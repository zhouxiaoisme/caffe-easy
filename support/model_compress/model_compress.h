


#pragma once

#include <support-common.h>


//infile�������ģ���ļ�·��
//fileOutPath�������ģ���ļ�·��
//upLevel��ѹ����ָ�꣬���ʵ�����ڶ�Ȩ�����任�����磺w' = ((int)(w * upLevel)) / upLevel;
//         ͨ��ȡֵ10000������ninȡֵ��300�������Լ�ʵ�ʲ����¾���
Caffe_API bool __stdcall model_compress(const char* infile, const char* fileOutPath, float upLevel);