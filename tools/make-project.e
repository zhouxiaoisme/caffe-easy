CNWTEPRGs壨
s 嫉溪Ｅ突盼s s s s s            <                                                                                                 s龌us 犆壕Ｅ位盼s s s s s            X                                                                                                                                  s跭;s 捞噪な协盼s s s s s          `
K`玧                                              R@揈`
� 	   _启动窗口   在程序启动后自动调入本窗口   �    
       �   �  <  �                           `�2   2   n  �                                                                      �                                                                          创建工程文件    �                         编辑框2  �(�    `   (                                                                       ���                        *             d                  ../windows-cpu        �                         save  *��      X   (                                                                                  保存k                         name  �*�0       �                                                                 峠   ( ( (# (% (& (' ($ (    �4  �9  �>  �>  1_  Nd  Ag  �4    vcxproj  �4  锘�<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" ToolsVersion="12.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Debug|Win32">
      <Configuration>Debug</Configuration>
      <Platform>Win32</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Debug|x64">
      <Configuration>Debug</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release|Win32">
      <Configuration>Release</Configuration>
      <Platform>Win32</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release|x64">
      <Configuration>Release</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  <PropertyGroup Label="Globals">
    <ProjectGuid>{9E094565-F231-4F61-B5BB-93E8108726DA}</ProjectGuid>
    <Keyword>Win32Proj</Keyword>
    <RootNamespace>ssd_call</RootNamespace>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|Win32'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <PlatformToolset>v120</PlatformToolset>
    <CharacterSet>Unicode</CharacterSet>
    <UseOfMfc>Static</UseOfMfc>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <PlatformToolset>v120</PlatformToolset>
    <CharacterSet>Unicode</CharacterSet>
    <UseOfMfc>Static</UseOfMfc>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|Win32'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>v120</PlatformToolset>
    <WholeProgramOptimization>true</WholeProgramOptimization>
    <CharacterSet>Unicode</CharacterSet>
    <UseOfMfc>Static</UseOfMfc>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>v120</PlatformToolset>
    <WholeProgramOptimization>true</WholeProgramOptimization>
    <CharacterSet>Unicode</CharacterSet>
    <UseOfMfc>Static</UseOfMfc>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" />
  <ImportGroup Label="ExtensionSettings">
  </ImportGroup>
  <ImportGroup Label="PropertySheets" Condition="'$(Configuration)|$(Platform)'=='Debug|Win32'">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
  </ImportGroup>
  <ImportGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'" Label="PropertySheets">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
  </ImportGroup>
  <ImportGroup Label="PropertySheets" Condition="'$(Configuration)|$(Platform)'=='Release|Win32'">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
  </ImportGroup>
  <ImportGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="PropertySheets">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
  </ImportGroup>
  <PropertyGroup Label="UserMacros" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|Win32'">
    <LinkIncremental>true</LinkIncremental>
    <IncludePath>../../include;../../3rd/include/protobuf;../../3rd/include/;../../3rd/include/openblas0.2.14.1;../../3rd/include/lmdb;../../include\caffe\proto;../../support\network;../../support\mtcnn;../../support\classification;../../support;../../support\train\lmdb;../../3rd/include/opencv2.4.10;../../3rd/include/opencv2.4.10/opencv;../../3rd/include/opencv2.4.10/opencv2;$(VC_IncludePath);$(WindowsSDK_IncludePath)</IncludePath>
    <LibraryPath>../../3rd/staticlib/boost/$(Platform)/;../../3rd/staticlib/gflags/$(Platform)/;../../3rd/staticlib/glog/$(Platform)/;../../3rd/staticlib/protobuf/$(Platform)/;../../3rd/staticlib/openblas/$(Platform)/;../../3rd/staticlib/lmdb/$(Platform)/;../../3rd/staticlib/opencv2.4.10/$(Platform)/vc$(PlatformToolsetVersion)/staticlib;$(VC_LibraryPath_x86);$(WindowsSDK_LibraryPath_x86)</LibraryPath>
    <OutDir>../../Build/cpu/$(Configuration)/$(Platform)/</OutDir>
    <IntDir>../../Build/cpu/outobj/$(ProjectName)/$(Configuration)/$(Platform)/</IntDir>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <LinkIncremental>true</LinkIncremental>
    <IncludePath>../../include;../../3rd/include/protobuf;../../3rd/include/;../../3rd/include/openblas0.2.14.1;../../3rd/include/lmdb;../../include\caffe\proto;../../support\network;../../support\mtcnn;../../support\classification;../../support;../../support\train\lmdb;../../3rd/include/opencv2.4.10;../../3rd/include/opencv2.4.10/opencv;../../3rd/include/opencv2.4.10/opencv2;$(VC_IncludePath);$(WindowsSDK_IncludePath)</IncludePath>
    <LibraryPath>../../3rd/staticlib/boost/$(Platform)/;../../3rd/staticlib/gflags/$(Platform)/;../../3rd/staticlib/glog/$(Platform)/;../../3rd/staticlib/protobuf/$(Platform)/;../../3rd/staticlib/openblas/$(Platform)/;../../3rd/staticlib/lmdb/$(Platform)/;../../3rd/staticlib/opencv2.4.10/$(Platform)/vc$(PlatformToolsetVersion)/staticlib;$(VC_LibraryPath_x64);$(WindowsSDK_LibraryPath_x64)</LibraryPath>
    <OutDir>../../Build/cpu/$(Configuration)/$(Platform)/</OutDir>
    <IntDir>../../Build/cpu/outobj/$(ProjectName)/$(Configuration)/$(Platform)/</IntDir>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|Win32'">
    <LinkIncremental>false</LinkIncremental>
    <IncludePath>../../include;../../3rd/include/protobuf;../../3rd/include/;../../3rd/include/openblas0.2.14.1;../../3rd/include/lmdb;../../include\caffe\proto;../../support\network;../../support\mtcnn;../../support\classification;../../support;../../support\train\lmdb;../../3rd/include/opencv2.4.10;../../3rd/include/opencv2.4.10/opencv;../../3rd/include/opencv2.4.10/opencv2;$(VC_IncludePath);$(WindowsSDK_IncludePath)</IncludePath>
    <LibraryPath>../../3rd/staticlib/boost/$(Platform)/;../../3rd/staticlib/gflags/$(Platform)/;../../3rd/staticlib/glog/$(Platform)/;../../3rd/staticlib/protobuf/$(Platform)/;../../3rd/staticlib/openblas/$(Platform)/;../../3rd/staticlib/lmdb/$(Platform)/;../../3rd/staticlib/opencv2.4.10/$(Platform)/vc$(PlatformToolsetVersion)/staticlib;$(VC_LibraryPath_x86);$(WindowsSDK_LibraryPath_x86)</LibraryPath>
    <OutDir>../../Build/cpu/$(Configuration)/$(Platform)/</OutDir>
    <IntDir>../../Build/cpu/outobj/$(ProjectName)/$(Configuration)/$(Platform)/</IntDir>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <LinkIncremental>false</LinkIncremental>
    <IncludePath>../../include;../../3rd/include/protobuf;../../3rd/include/;../../3rd/include/openblas0.2.14.1;../../3rd/include/lmdb;../../include\caffe\proto;../../support\network;../../support\mtcnn;../../support\classification;../../support;../../support\train\lmdb;../../3rd/include/opencv2.4.10;../../3rd/include/opencv2.4.10/opencv;../../3rd/include/opencv2.4.10/opencv2;$(VC_IncludePath);$(WindowsSDK_IncludePath)</IncludePath>
    <LibraryPath>../../3rd/staticlib/boost/$(Platform)/;../../3rd/staticlib/gflags/$(Platform)/;../../3rd/staticlib/glog/$(Platform)/;../../3rd/staticlib/protobuf/$(Platform)/;../../3rd/staticlib/openblas/$(Platform)/;../../3rd/staticlib/lmdb/$(Platform)/;../../3rd/staticlib/opencv2.4.10/$(Platform)/vc$(PlatformToolsetVersion)/staticlib;$(VC_LibraryPath_x64);$(WindowsSDK_LibraryPath_x64)</LibraryPath>
    <OutDir>../../Build/cpu/$(Configuration)/$(Platform)/</OutDir>
    <IntDir>../../Build/cpu/outobj/$(ProjectName)/$(Configuration)/$(Platform)/</IntDir>
  </PropertyGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|Win32'">
    <ClCompile>
      <PrecompiledHeader>
      </PrecompiledHeader>
      <WarningLevel>Level3</WarningLevel>
      <Optimization>Disabled</Optimization>
      <PreprocessorDefinitions>WIN32;_CRT_SECURE_NO_DEPRECATE;USE_OPENCV;_SCL_SECURE_NO_WARNINGS;_CRT_SECURE_NO_WARNINGS;_CRT_NONSTDC_NO_DEPRECATE;CPU_ONLY;_DEBUG;_CONSOLE;_LIB;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <SDLCheck>true</SDLCheck>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <IgnoreSpecificDefaultLibraries>MSVCRTD.lib</IgnoreSpecificDefaultLibraries>
    </Link>
    <PostBuildEvent>
      <Command>
      </Command>
    </PostBuildEvent>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <ClCompile>
      <PrecompiledHeader>
      </PrecompiledHeader>
      <WarningLevel>Level3</WarningLevel>
      <Optimization>Disabled</Optimization>
      <PreprocessorDefinitions>USE_LMDB;USE_LMDB;WIN32;_CRT_SECURE_NO_DEPRECATE;USE_OPENCV;_SCL_SECURE_NO_WARNINGS;_CRT_SECURE_NO_WARNINGS;_CRT_NONSTDC_NO_DEPRECATE;CPU_ONLY;_DEBUG;_CONSOLE;_LIB;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <SDLCheck>true</SDLCheck>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <IgnoreSpecificDefaultLibraries>MSVCRTD.lib</IgnoreSpecificDefaultLibraries>
    </Link>
    <PostBuildEvent>
      <Command>
      </Command>
    </PostBuildEvent>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|Win32'">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <PrecompiledHeader>
      </PrecompiledHeader>
      <Optimization>MaxSpeed</Optimization>
      <FunctionLevelLinking>true</FunctionLevelLinking>
      <IntrinsicFunctions>true</IntrinsicFunctions>
      <PreprocessorDefinitions>WIN32;_CRT_SECURE_NO_DEPRECATE;USE_OPENCV;_SCL_SECURE_NO_WARNINGS;_CRT_SECURE_NO_WARNINGS;_CRT_NONSTDC_NO_DEPRECATE;CPU_ONLY;NDEBUG;_CONSOLE;_LIB;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <SDLCheck>true</SDLCheck>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
      <IgnoreSpecificDefaultLibraries>MSVCRT.lib</IgnoreSpecificDefaultLibraries>
    </Link>
    <PostBuildEvent>
      <Command>
      </Command>
    </PostBuildEvent>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <PrecompiledHeader>
      </PrecompiledHeader>
      <Optimization>MaxSpeed</Optimization>
      <FunctionLevelLinking>true</FunctionLevelLinking>
      <IntrinsicFunctions>true</IntrinsicFunctions>
      <PreprocessorDefinitions>USE_LMDB;USE_LMDB;WIN32;_CRT_SECURE_NO_DEPRECATE;USE_OPENCV;_SCL_SECURE_NO_WARNINGS;_CRT_SECURE_NO_WARNINGS;_CRT_NONSTDC_NO_DEPRECATE;CPU_ONLY;NDEBUG;_CONSOLE;_LIB;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <SDLCheck>true</SDLCheck>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
      <IgnoreSpecificDefaultLibraries>MSVCRT.lib</IgnoreSpecificDefaultLibraries>
    </Link>
    <PostBuildEvent>
      <Command>
      </Command>
    </PostBuildEvent>
  </ItemDefinitionGroup>
  <ItemGroup>
    <ProjectReference Include="..\libcaffe\windows-cpu.vcxproj">
      <Project>{c140902d-9ab4-4c4e-8684-9077c476cdb7}</Project>
      <Private>true</Private>
      <ReferenceOutputAssembly>true</ReferenceOutputAssembly>
      <CopyLocalSatelliteAssemblies>true</CopyLocalSatelliteAssemblies>
      <LinkLibraryDependencies>true</LinkLibraryDependencies>
      <UseLibraryDependencyInputs>true</UseLibraryDependencyInputs>
    </ProjectReference>
  </ItemGroup>
  <ItemGroup>
    <ClCompile Include="..\..\support\classification\classification.cpp" />
    <ClCompile Include="..\..\support\ssd\pa_draw.cpp" />
    <ClCompile Include="..\..\support\ssd\ssd_call.cpp" />
  </ItemGroup>
  <ItemGroup>
    <ClInclude Include="..\..\support\classification\classification-c.h" />
    <ClInclude Include="..\..\support\classification\classification.h" />
    <ClInclude Include="..\..\support\ssd\pa_draw.h" />
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" />
  <ImportGroup Label="ExtensionTargets">
  </ImportGroup>
</Project>    vcxprojfilters  �  锘�<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup>
    <Filter Include="婧愭枃浠�">
      <UniqueIdentifier>{4FC737F1-C7A5-4376-A066-2A32D752A2FF}</UniqueIdentifier>
      <Extensions>cpp;c;cc;cxx;def;odl;idl;hpj;bat;asm;asmx</Extensions>
    </Filter>
    <Filter Include="澶存枃浠�">
      <UniqueIdentifier>{56d820d5-46b2-4ce9-ba0c-fd73bcc18aef}</UniqueIdentifier>
    </Filter>
  </ItemGroup>
  <ItemGroup>
    <ClCompile Include="..\..\support\ssd\ssd_call.cpp">
      <Filter>婧愭枃浠�</Filter>
    </ClCompile>
    <ClCompile Include="..\..\support\classification\classification.cpp">
      <Filter>婧愭枃浠�</Filter>
    </ClCompile>
    <ClCompile Include="..\..\support\ssd\pa_draw.cpp">
      <Filter>婧愭枃浠�</Filter>
    </ClCompile>
  </ItemGroup>
  <ItemGroup>
    <ClInclude Include="..\..\support\classification\classification.h">
      <Filter>澶存枃浠�</Filter>
    </ClInclude>
    <ClInclude Include="..\..\support\classification\classification-c.h">
      <Filter>澶存枃浠�</Filter>
    </ClInclude>
    <ClInclude Include="..\..\support\ssd\pa_draw.h">
      <Filter>澶存枃浠�</Filter>
    </ClInclude>
  </ItemGroup>
</Project>�    vcxprojuser  �  锘�<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="12.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|Win32'">
    <LocalDebuggerWorkingDirectory>../../Build/cpu/$(Configuration)/$(Platform)/</LocalDebuggerWorkingDirectory>
    <DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|Win32'">
    <LocalDebuggerWorkingDirectory>../../Build/cpu/$(Configuration)/$(Platform)/</LocalDebuggerWorkingDirectory>
    <DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <LocalDebuggerWorkingDirectory>../../Build/cpu/$(Configuration)/$(Platform)/</LocalDebuggerWorkingDirectory>
    <DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <LocalDebuggerWorkingDirectory>../../Build/cpu/$(Configuration)/$(Platform)/</LocalDebuggerWorkingDirectory>
    <DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>
  </PropertyGroup>
</Project>           n     vcxproj_gpu  [   锘�<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" ToolsVersion="12.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Debug|x64">
      <Configuration>Debug</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release|x64">
      <Configuration>Release</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  <PropertyGroup Label="Globals">
    <ProjectGuid>{9E094565-F231-4F61-B5BB-93E8108726DA}</ProjectGuid>
    <Keyword>Win32Proj</Keyword>
    <RootNamespace>ssd_call</RootNamespace>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <PlatformToolset>v120</PlatformToolset>
    <CharacterSet>Unicode</CharacterSet>
    <UseOfMfc>Static</UseOfMfc>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>v120</PlatformToolset>
    <WholeProgramOptimization>true</WholeProgramOptimization>
    <CharacterSet>Unicode</CharacterSet>
    <UseOfMfc>Static</UseOfMfc>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" />
  <ImportGroup Label="ExtensionSettings">
    <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA $(CC_CUDA_VERSION).props" />
  </ImportGroup>
  <ImportGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'" Label="PropertySheets">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
  </ImportGroup>
  <ImportGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="PropertySheets">
    <Import Project="$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
  </ImportGroup>
  <PropertyGroup Label="UserMacros" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <LinkIncremental>true</LinkIncremental>
    <IncludePath>../../include;../../3rd/include/protobuf;../../3rd/include/;../../3rd/include/openblas0.2.14.1;../../include\caffe\proto;../../support\network;../../support\mtcnn;../../support\classification;../../support;../../support\train\lmdb;../../3rd/include/opencv2.4.10;../../3rd/include/opencv2.4.10/opencv;../../3rd/include/opencv2.4.10/opencv2;$(VC_IncludePath);$(WindowsSDK_IncludePath)</IncludePath>
    <LibraryPath>../../3rd/staticlib/boost/$(Platform)/;../../3rd/staticlib/gflags/$(Platform)/;../../3rd/staticlib/glog/$(Platform)/;../../3rd/staticlib/protobuf/$(Platform)/;../../3rd/staticlib/openblas/$(Platform)/;../../3rd/staticlib/opencv2.4.10/$(Platform)/vc$(PlatformToolsetVersion)/staticlib;../../3rd/staticlib/lmdb/$(Platform)/;../../Build/gpu_cuda$(CC_CUDA_VERSION)/$(Configuration)/$(Platform)/;$(VC_LibraryPath_x64);$(WindowsSDK_LibraryPath_x64)</LibraryPath>
    <OutDir>../../Build/gpu_cuda$(CC_CUDA_VERSION)/$(Configuration)/$(Platform)/</OutDir>
    <IntDir>../../Build/gpu_cuda$(CC_CUDA_VERSION)/outobj/$(ProjectName)/$(Configuration)/$(Platform)/</IntDir>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <LinkIncremental>false</LinkIncremental>
    <IncludePath>../../include;../../3rd/include/protobuf;../../3rd/include/;../../3rd/include/openblas0.2.14.1;../../include\caffe\proto;../../support\network;../../support\mtcnn;../../support\classification;../../support;../../support\train\lmdb;../../3rd/include/opencv2.4.10;../../3rd/include/opencv2.4.10/opencv;../../3rd/include/opencv2.4.10/opencv2;$(VC_IncludePath);$(WindowsSDK_IncludePath)</IncludePath>
    <LibraryPath>../../3rd/staticlib/boost/$(Platform)/;../../3rd/staticlib/gflags/$(Platform)/;../../3rd/staticlib/glog/$(Platform)/;../../3rd/staticlib/protobuf/$(Platform)/;../../3rd/staticlib/openblas/$(Platform)/;../../3rd/staticlib/opencv2.4.10/$(Platform)/vc$(PlatformToolsetVersion)/staticlib;../../3rd/staticlib/lmdb/$(Platform)/;../../Build/gpu_cuda$(CC_CUDA_VERSION)/$(Configuration)/$(Platform)/;$(VC_LibraryPath_x64);$(WindowsSDK_LibraryPath_x64)</LibraryPath>
    <OutDir>../../Build/gpu_cuda$(CC_CUDA_VERSION)/$(Configuration)/$(Platform)/</OutDir>
    <IntDir>../../Build/gpu_cuda$(CC_CUDA_VERSION)/outobj/$(ProjectName)/$(Configuration)/$(Platform)/</IntDir>
  </PropertyGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <ClCompile>
      <PrecompiledHeader>
      </PrecompiledHeader>
      <WarningLevel>Level3</WarningLevel>
      <Optimization>Disabled</Optimization>
      <PreprocessorDefinitions>USE_LMDB;WIN32;_CRT_SECURE_NO_DEPRECATE;USE_OPENCV;_SCL_SECURE_NO_WARNINGS;_CRT_SECURE_NO_WARNINGS;_CRT_NONSTDC_NO_DEPRECATE;_DEBUG;_CONSOLE;_LIB;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <SDLCheck>true</SDLCheck>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <IgnoreSpecificDefaultLibraries>MSVCRTD.lib</IgnoreSpecificDefaultLibraries>
      <AdditionalDependencies>libcaffe.lib</AdditionalDependencies>
    </Link>
    <PostBuildEvent>
      <Command>
      </Command>
    </PostBuildEvent>
    <CudaCompile>
      <CodeGeneration>compute_30,sm_30</CodeGeneration>
    </CudaCompile>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <PrecompiledHeader>
      </PrecompiledHeader>
      <Optimization>MaxSpeed</Optimization>
      <FunctionLevelLinking>true</FunctionLevelLinking>
      <IntrinsicFunctions>true</IntrinsicFunctions>
      <PreprocessorDefinitions>USE_LMDB;WIN32;_CRT_SECURE_NO_DEPRECATE;USE_OPENCV;_SCL_SECURE_NO_WARNINGS;_CRT_SECURE_NO_WARNINGS;_CRT_NONSTDC_NO_DEPRECATE;NDEBUG;_CONSOLE;_LIB;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <SDLCheck>true</SDLCheck>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
      <IgnoreSpecificDefaultLibraries>MSVCRT.lib</IgnoreSpecificDefaultLibraries>
      <AdditionalDependencies>libcaffe.lib</AdditionalDependencies>
    </Link>
    <PostBuildEvent>
      <Command>
      </Command>
    </PostBuildEvent>
    <CudaCompile>
      <CodeGeneration>compute_30,sm_30</CodeGeneration>
    </CudaCompile>
  </ItemDefinitionGroup>
  <ItemGroup>
    <ProjectReference Include="..\libcaffe\windows-gpu.vcxproj">
      <Project>{c140902d-9ab4-4c4e-8684-9077c476cdb7}</Project>
      <Private>false</Private>
      <ReferenceOutputAssembly>true</ReferenceOutputAssembly>
      <CopyLocalSatelliteAssemblies>true</CopyLocalSatelliteAssemblies>
      <LinkLibraryDependencies>true</LinkLibraryDependencies>
      <UseLibraryDependencyInputs>true</UseLibraryDependencyInputs>
    </ProjectReference>
  </ItemGroup>
  <ItemGroup>
    <ClCompile Include="..\..\support\classification\classification.cpp" />
    <ClCompile Include="..\..\support\ssd\pa_draw.cpp" />
    <ClCompile Include="..\..\support\ssd\ssd_call.cpp" />
  </ItemGroup>
  <ItemGroup>
    <ClInclude Include="..\..\support\classification\classification-c.h" />
    <ClInclude Include="..\..\support\classification\classification.h" />
    <ClInclude Include="..\..\support\ssd\pa_draw.h" />
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" />
  <ImportGroup Label="ExtensionTargets">
    <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA $(CC_CUDA_VERSION).targets" />
  </ImportGroup>
</Project>    vcxprojfilters_gpu  �  锘�<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup>
    <Filter Include="婧愭枃浠�">
      <UniqueIdentifier>{4FC737F1-C7A5-4376-A066-2A32D752A2FF}</UniqueIdentifier>
      <Extensions>cpp;c;cc;cxx;def;odl;idl;hpj;bat;asm;asmx</Extensions>
    </Filter>
    <Filter Include="澶存枃浠�">
      <UniqueIdentifier>{56d820d5-46b2-4ce9-ba0c-fd73bcc18aef}</UniqueIdentifier>
    </Filter>
  </ItemGroup>
  <ItemGroup>
    <ClCompile Include="..\..\support\ssd\ssd_call.cpp">
      <Filter>婧愭枃浠�</Filter>
    </ClCompile>
    <ClCompile Include="..\..\support\classification\classification.cpp">
      <Filter>婧愭枃浠�</Filter>
    </ClCompile>
    <ClCompile Include="..\..\support\ssd\pa_draw.cpp">
      <Filter>婧愭枃浠�</Filter>
    </ClCompile>
  </ItemGroup>
  <ItemGroup>
    <ClInclude Include="..\..\support\classification\classification.h">
      <Filter>澶存枃浠�</Filter>
    </ClInclude>
    <ClInclude Include="..\..\support\classification\classification-c.h">
      <Filter>澶存枃浠�</Filter>
    </ClInclude>
    <ClInclude Include="..\..\support\ssd\pa_draw.h">
      <Filter>澶存枃浠�</Filter>
    </ClInclude>
  </ItemGroup>
</Project>�    vcxprojuser_gpu  �  锘�<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="12.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <LocalDebuggerWorkingDirectory>../../Build/gpu_cuda$(CC_CUDA_VERSION)/$(Configuration)/$(Platform)/</LocalDebuggerWorkingDirectory>
    <DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <LocalDebuggerWorkingDirectory>../../Build/gpu_cuda$(CC_CUDA_VERSION)/$(Configuration)/$(Platform)/</LocalDebuggerWorkingDirectory>
    <DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>
  </PropertyGroup>
</Project>               s骡�s 捞与盼s s s s s s          擲                                          7  ?�   �             9   krnlnd09f2340818511d396f6aaf844c7e32546系统核心支持库                    鄴E R       窗口程序集1                       扙 慐            _save_被单击                           3                    "       F   j ��          68 9	     78
 9	     7j    ��          6            writeTo       2    % %            �   proj       �   ndir     4    % %            �   project       �   dir  D       I   d   �   �   �  �    Z  �  �  �  �  �    [  �  	   c   �      l      +   9   [   �   �     �  �  �    /  6  ~  �  �  �    �  �  �    0  7    �  �      �   T  �  �  U  �  �  j4               68 %7!               68 %7   / 8 %7   / j�               68 %7mn               6!'               6!R               68 %7   gpu         鹂j4               68 %7% (j4               68 %7!n               68 %7!f               60   <RootNamespace>convert_imageset</RootNamespace> !f               6!               6   <RootNamespace> 8 %7   </RootNamespace> j    ��          6j�               6!               68 %78 %7	   .vcxproj 8 %7j�               6!               68 %78 %7   .vcxproj.filters & (j�               6!               68 %78 %7   .vcxproj.user ' (Soj    ��          6j4               68 %7 (j4               68 %7!n               68 %7!f               60   <RootNamespace>convert_imageset</RootNamespace> !f               6!               6   <RootNamespace> 8 %7   </RootNamespace> j    ��          6j�               6!               68 %78 %7	   .vcxproj 8 %7j�               6!               68 %78 %7   .vcxproj.filters  (j�               6!               68 %78 %7   .vcxproj.user  (Ttj    ��          6                                                        s垭}Ds 栗楼罚佛盼s s s s s                                                               sdIKs 躺吵恭墩ｒ匙s s s s s         ����                                                                        ����s翞s 	锣荡Ｅ苹盼;s 	s 	s 	s 	s         枀墎[                                                    �              �             R     I  �            �        s坌#s 
栓茔Ｅ呕盼;s 
s 
s 
s 
s          @                                            R      ss s                                 	                                                       