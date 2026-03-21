
Skip to content
yasirkula
UnityRuntimeInspector
Repository navigation
Code
Issues
6
 (6)
Pull requests
3
 (3)
Agents
Discussions
Security
Insights
Owner avatar
UnityRuntimeInspector
Public
yasirkula/UnityRuntimeInspector
Go to file
t
Name		
yasirkula
yasirkula
Simplified Package Manager installation instruction
f8755d6
 · 
10 months ago
.github
Simplified Package Manager installation instruction
10 months ago
Plugins
- Updated Unity version to 2021.3.41f1
last year
LICENSE.txt
Added Package Manager support
6 years ago
LICENSE.txt.meta
Added Package Manager support
6 years ago
Plugins.meta
Added Package Manager support
6 years ago
package.json
- Updated Unity version to 2021.3.41f1
last year
package.json.meta
Added Package Manager support
6 years ago
Repository files navigation
README
Contributing
MIT license
Runtime Inspector & Hierarchy for Unity 3D
screenshot

Available on Asset Store: https://assetstore.unity.com/packages/tools/gui/runtime-inspector-hierarchy-111349
可在 Asset Store 获取： https://assetstore.unity.com/packages/tools/gui/runtime-inspector-hierarchy-111349

Forum Thread: https://forum.unity.com/threads/runtime-inspector-and-hierarchy-open-source.501220/
论坛帖子： https://forum.unity.com/threads/runtime-inspector-and-hierarchy-open-source.501220/

Discord: https://discord.gg/UJJt549AaV
Discord： https://discord.gg/UJJt549AaV

GitHub Sponsors ☕  GitHub 赞助商☕

A. ABOUT  A. 关于
This is a simple yet powerful runtime Inspector and Hierarchy solution for Unity 3D that should work on pretty much any platform that Unity supports, including mobile platforms.
这是一个简单而强大的 Unity 3D 运行时检查器和层级解决方案，几乎可以在 Unity 支持的任何平台上运行，包括移动平台。

B. LICENSE  B. 许可证
Runtime Inspector & Hierarchy is licensed under the MIT License (Asset Store version is governed by the Asset Store EULA). Please note that this asset uses an external asset which is licensed under the BSD 3-Clause License.
Runtime Inspector & Hierarchy 采用 MIT 许可证授权（ Asset Store 版本受 Asset Store EULA 约束）。请注意，此资源使用了外部资源，该外部资源采用 BSD 3-Clause 许可证授权。

C. INSTALLATION  C. 安装
There are 5 ways to install this plugin:
安装此插件有 5 种方法：

import RuntimeInspector.unitypackage via Assets-Import Package
通过 Assets-Import Package 导入 RuntimeInspector.unitypackage
clone/download this repository and move the Plugins folder to your Unity project's Assets folder
克隆/ 下载此存储库，并将 Plugins 文件夹移动到 Unity 项目的 Assets 文件夹中。
import it from Asset Store
从 Asset Store 导入
(via Package Manager) click the + button and install the package from the following git URL:
（通过软件包管理器） 点击“+”按钮，然后从以下 git URL 安装软件包：
https://github.com/yasirkula/UnityRuntimeInspector.git
(via OpenUPM) after installing openupm-cli, run the following command:
（通过 OpenUPM ） 安装 openupm-cli 后，运行以下命令：
openupm add com.yasirkula.runtimeinspector
FAQ  常问问题
New Input System isn't supported on Unity 2019.2.5 or earlier
Unity 2019.2.5 或更早版本不支持新的输入系统。
Add ENABLE_INPUT_SYSTEM compiler directive to Player Settings/Scripting Define Symbols (these symbols are platform specific, so if you change the active platform later, you'll have to add the compiler directive again).
将 ENABLE_INPUT_SYSTEM 编译器指令添加到玩家设置/脚本定义符号中 （这些符号是平台特定的，因此如果您稍后更改活动平台，则必须再次添加编译器指令）。

"Unity.InputSystem" assembly can't be resolved on Unity 2018.4 or earlier
在 Unity 2018.4 或更早版本中无法解析“Unity.InputSystem”程序集。
Remove Unity.InputSystem assembly from RuntimeInspector.Runtime Assembly Definition File's Assembly Definition References list.
从 RuntimeInspector.Runtime 程序集定义文件的程序集定义引用列表中移除 Unity.InputSystem 程序集。

D. HOW TO  D. 如何
To use the hierarchy in your scene, drag&drop the RuntimeHierarchy prefab to your canvas
要在场景中使用层级结构，请将 RuntimeHierarchy 预制件拖放到画布上。
To use the inspector in your scene, drag&drop the RuntimeInspector prefab to your canvas
要在场景中使用检查器，请将 RuntimeInspector 预制件拖放到画布上。
You can connect the inspector to the hierarchy so that whenever the selection in the hierarchy changes, inspector inspects the newly selected object. To do this, assign the inspector to the Connected Inspector property of the hierarchy.
您可以将检查器连接到层级结构，这样，每当层级结构中的选择发生变化时，检查器都会检查新选择的对象。为此，请将检查器分配给层级结构的 “已连接检查器” 属性。

You can also connect the hierarchy to the inspector so that whenever an object reference in the inspector is highlighted, the selection in hierarchy is updated. To do this, assign the hierarchy to the Connected Hierarchy property of the inspector.
您还可以将层级结构连接到检查器，这样，每当检查器中的对象引用被选中时，层级结构中的选择也会随之更新。为此，请将层级结构分配给检查器的 “已连接层级结构” 属性。

Note that these connections are one-directional, meaning that assigning the inspector to the hierarchy will not automatically assign the hierarchy to the inspector or vice versa. Also note that the inspector and the hierarchy are not singletons and therefore, you can have several instances of them in your scene at a time with different configurations.
请注意，这些连接是单向的 ，这意味着将检查器分配给层级结构不会自动将层级结构分配给检查器，反之亦然。另请注意，检查器和层级结构都不是单例，因此，您可以在场景中同时拥有多个具有不同配置的检查器和层级结构实例。

E. FEATURES  E. 特点
Both panels are heavily optimized in terms of GC in order not to cause any unnecessary allocations. By default, both the inspector and the hierarchy are refreshed 4 times a second to reflect any changes to their user interface almost immediately. Each refresh of the inspector generates some garbage for GC since most of the time, the inspected object has variables of value types. These variables are boxed when accessed via reflection and this boxing creates some unavoidable garbage. However, this process can be greatly optimized by increasing the Refresh Interval of the inspector and/or the hierarchy
这两个面板都针对垃圾回收进行了深度优化，以避免不必要的内存分配。默认情况下，检查器和层级视图每秒刷新 4 次，以便几乎立即反映用户界面的任何更改。每次刷新检查器都会产生一些垃圾需要进行垃圾回收，因为大多数情况下，被检查的对象都包含值类型的变量。这些变量在通过反射访问时会被装箱 ，而这种装箱操作会产生一些不可避免的垃圾。但是，可以通过增加检查器和/或层级视图的刷新间隔来显著优化这个过程。
Includes a built-in color picker and a reference picker:
内置颜色选择器和参考选择器：
screenshot

Visual appearance of the inspector and the hierarchy can be tweaked by changing their Skin. There are two premade skins included in the Skins directory: LightSkin and DarkSkin. You can create your own skins using the Assets-Create-yasirkula-RuntimeInspector-UI Skin context menu
可以通过更改皮肤来调整检查器及其层级的视觉外观。Skins 目录中包含两个预制皮肤： LightSkin 和 DarkSkin 。您可以使用 Assets-Create-yasirkula-RuntimeInspector-UI Skin 上下文菜单创建自己的皮肤。
screenshot

The hierarchy supports multi-selection:
该层级结构支持多选：
screenshot

E.1. INSPECTOR  E.1. 检查员
screenshot

RuntimeInspector works similar to the editor Inspector. It can expose commonly used Unity types out-of-the-box, as well as custom classes and structs that are marked with System.Serializable attribute. 1-dimensional arrays and generic Lists are also supported.
RuntimeInspector 的工作方式与编辑器 Inspector 类似。它可以开箱即用地公开常用的 Unity 类型，以及使用 System.Serializable 属性标记的自定义类和结构体。此外，它还支持一维数组和泛型列表。

Refresh Interval: as the name suggests, this is the refresh interval of the inspector. At each refresh, values of all the exposed fields and properties are refreshed. This generates some garbage for boxed value types (unavoidable) and thus, increasing this value even slightly should help with GC a lot
刷新间隔 ：顾名思义，这是检查器的刷新间隔。每次刷新时，所有公开字段和属性的值都会被刷新。这会为装箱值类型生成一些垃圾数据（这是不可避免的），因此，即使稍微增加此值也能显著改善垃圾回收。
Expose Fields: determines which fields of the inspected object should be exposed: None, Serializable Only or All
公开字段 ：确定应公开被检查对象的哪些字段： 无 、 仅可序列化字段或全部
Expose Properties: determines which properties of the inspected object should be exposed
公开属性 ：确定应公开被检查对象的哪些属性。
Array Indices Start At One: when enabled, exposed arrays and lists start their indices at 1 instead of 0 (just a visual change)
数组索引从 1 开始 ：启用后，公开的数组和列表的索引将从 1 开始而不是 0（只是视觉上的变化）。
Use Title Case Naming: when enabled, variable names are displayed in title case format (e.g. m_myVariable becomes My Variable)
使用首字母大写命名 ：启用后，变量名将以首字母大写格式显示（例如， m_myVariable 变为 My Variable ）。
Show Add Component Button: when enabled, Add Component button will appear while inspecting a GameObject
显示“添加组件”按钮 ：启用后，在检查游戏对象时将显示 “添加组件” 按钮。
Show Remove Component Button: when enabled, Remove Component button will appear under inspected components
显示“移除组件”按钮 ：启用后， “移除组件” 按钮将显示在已检查组件下方。
Show Inspect Reference Button: when enabled, ObjectReferenceFields will show an arrow next to the selected Object reference. When that arrow is clicked, inspector will automatically inspect that Object
显示“检查引用”按钮 ：启用后， “对象引用字段 ”将在选定的对象引用旁边显示一个箭头。单击该箭头，检查器将自动检查该对象。
Show Tooltips: when enabled, hovering over a variable's name for a while will show a tooltip displaying the variable's name. Can be useful for variables whose names are partially obscured
显示工具提示 ：启用后，将鼠标悬停在变量名称上片刻，即可显示包含变量名称的工具提示。这对于名称部分被遮挡的变量非常有用。
Tooltip Delay: determines how long the cursor should remain static over a variable's name before the tooltip appears. Has no effect if Show Tooltips is disabled
工具提示延迟 ：决定光标在变量名上停留多长时间后才会显示工具提示。如果 “显示工具提示” 已禁用，则此设置无效。
Nest Limit: imagine exposing a linked list. This variable defines how many nodes you can expose in the inspector starting from the initial node until the inspector stops exposing any further nodes
嵌套限制 ：想象一下显示一个链表。此变量定义了从初始节点开始，检查器可以显示多少个节点，直到检查器停止显示任何后续节点为止。
Inspected Object Header Visibility: if the inspected object has a collapsible header, determines that header's visibility
被检查对象头部可见性 ：如果被检查对象具有可折叠的头部，则确定该头部的可见性
Pool Capacity: the UI elements are pooled to avoid unnecessary Instantiate and Destroy calls. This value defines the pool capacity for each of the UI elements individually. On standalone platforms, you can increase this value for better performance
池容量 ：UI 元素被放入池中，以避免不必要的实例化和销毁调用。此值定义了每个 UI 元素的池容量。在独立平台上，您可以增加此值以提高性能。
Settings: an array of settings for the inspector. A new settings asset can be created using the Assets-Create-yasirkula-RuntimeInspector-Settings context menu. A setting asset stores 4 different things:
设置 ：检查器设置的数组。可以使用 “资源”-“创建”-“yasirkula”-“运行时检查器”-“设置” 上下文菜单创建新的设置资源。设置资源存储 4 种不同的内容：
Standard Drawers and Reference Drawers: a drawer is a prefab used to expose a single variable in the inspector. For variables that extend UnityEngine.Object, a reference drawer is created and for other variables, a standard drawer is created
标准抽屉和引用抽屉 ：抽屉是一种预制件，用于在检视面板中显示单个变量。对于继承自 UnityEngine.Object 的变量，会创建一个引用抽屉；对于其他变量，会创建一个标准抽屉。
While searching for a suitable drawer for a variable, the corresponding drawers list is traversed from bottom to top until a drawer that supports that variable type is found. If such a drawer is not found, that variable is not exposed
在查找适合变量的抽屉时，会从下到上遍历相应的抽屉列表，直到找到支持该变量类型的抽屉为止。如果找不到这样的抽屉，则不会公开该变量。
Hidden Variables: allows you to hide some variables from the inspector for a given type and all the types that extend/implement it. You can enter asterisk character (*) to hide all the variables for that type
隐藏变量 ：允许您对给定类型及其所有扩展/实现类型隐藏某些变量。您可以使用星号 (*) 隐藏该类型的所有变量。
Exposed Variables: allows you to expose (counter) some hidden variables. A variable goes through a number of filters before it is exposed:
公开变量 ：允许您公开（显示）一些隐藏变量。变量在公开之前会经过一系列过滤器：
Its Type must be serializable
它的类型必须是可序列化的。
It must not have a System.Obsolete, System.NonSerialized or HideInInspector attribute
它不能具有 System.Obsolete 、 System.NonSerialized 或 HideInInspector 属性。
If it is in Exposed Variables, it is exposed
如果它位于 “公开变量” 中，则表示它已公开
It must not be in Hidden Variables
它不能位于隐藏变量中。
it must pass the Expose Fields and Expose Properties filters
它必须通过 “公开字段” 和 “公开属性” 筛选器。
So, to expose only a specific set of variables for a given type, you can hide all of its variables by entering an asterisk to its Hidden Variables and then entering the set of exposed variables to its Exposed Variables
因此，要仅公开给定类型的一组特定变量，您可以通过在其 “隐藏变量” 中输入星号来隐藏其所有变量，然后在其 “公开变量” 中输入要公开的变量集。
While changing the inspector's settings, you are advised not to touch InternalSettings; instead create a separate Settings asset and add it to the Settings array of the inspector. Otherwise, when InternalSettings is changed in an update, your settings might get overridden.
修改检查器设置时，建议不要直接修改 InternalSettings ；而是创建一个单独的 Settings 资源，并将其添加到检查器的 Settings 数组中。否则，当 InternalSettings 在更新过程中发生更改时，您的设置可能会被覆盖。

E.2. HIERARCHY  E.2. 层级结构
screenshot

RuntimeHierarchy simply exposes the objects in your scenes to the user interface. In addition to exposing the currently active Unity scenes in the hierarchy, you can also expose a specific set of objects under what is called a pseudo-scene in the hierarchy. Pseudo-scenes can help you categorize the objects in your scene. Adding/removing objects to/from pseudo-scenes is only possible via the scripting API and helper components.
RuntimeHierarchy 的作用是将场景中的对象暴露给用户界面。除了在层级视图中显示当前活动的 Unity 场景之外，你还可以将一组特定的对象（称为伪场景） 暴露在层级视图中。伪场景可以帮助你对场景中的对象进行分类。向伪场景添加/从伪场景中移除对象只能通过脚本 API 和辅助组件来实现。

Refresh Interval: the refresh interval of the hierarchy. At each refresh, the destroyed objects are removed from the hierarchy while newly created objects are added to the hierarchy. Sibling indices of the objects are also synced with the Unity Hierarchy at each refresh
刷新间隔 ：层级结构的刷新间隔。每次刷新时，已销毁的对象会从层级结构中移除，而新创建的对象则会添加到层级结构中。每次刷新时，对象的兄弟索引也会与 Unity 层级结构同步。
Object Names Refresh Interval: accessing GameObject.name property generates garbage. Therefore, names of objects in the hierarchy are not synced at each Refresh Interval but rather at each Object Names Refresh Interval to help avoid excessive garbage
对象名称刷新间隔 ：访问 GameObject.name 属性会产生垃圾数据。因此，层级结构中的对象名称并非在每个刷新间隔同步，而是在每个对象名称刷新间隔同步， 以避免产生过多垃圾数据。
Search Refresh Interval: the refresh interval for the search results. At each refresh, each GameObject's name is checked to see if it matches the searched term, so this process will generate some garbage
搜索刷新间隔 ：搜索结果的刷新间隔。每次刷新时，系统都会检查每个游戏对象的名称是否与搜索词匹配，因此此过程会产生一些垃圾数据。
Allow Multi Selection: when disabled, only a single Transform can be selected in the hierarchy
允许多选 ：禁用此选项后，层次结构中只能选择一个变换。
Expose Unity Scenes: when disabled, Unity scenes are not exposed in the hierarchy. This is useful when you want to use the hierarchy solely for pseudo-scenes
公开 Unity 场景 ：禁用此选项后，Unity 场景将不会在层级视图中显示。如果您只想将层级视图用于伪场景，这将非常有用。
Exposed Unity Scenes Subset: specifies the scenes that are exposed in the hierarchy by their name. When empty, all scenes are exposed
公开的 Unity 场景子集 ：指定在层级结构中按名称公开的场景。如果为空，则公开所有场景。
Expose Dont Destroy On Load Scene: when enabled, DontDesroyOnLoad objects will be exposed in the hierarchy
公开加载时不销毁场景 ：启用后， DontDesroyOnLoad 对象将在层级视图中公开。
Pseudo Scenes Order: the order of the pseudo-scenes from top to bottom in the hierarchy. Note that entering a pseudo-scene here does not automatically create it when the application starts. Pseudo-scenes can be created via the scripting API only
伪场景顺序 ：伪场景在层级结构中从上到下的顺序。请注意，在此处输入伪场景并不会在应用程序启动时自动创建它。伪场景只能通过脚本 API 创建。
Pointer Long Press Action: determines what will happen when an object is clicked and then held for a while:
指针长按操作 ：决定点击并按住某个对象一段时间后会发生什么：
None: nothing ¯\_(ツ)_/¯
无 ：什么都没有 ¯\_(ツ)_/¯
Create Dragged Reference Item: creates a dragged reference item that can be dropped onto a reference drawer in the inspector to assign the held object(s) to that variable (similar to Unity's drag&drop reference assignment)
创建拖拽引用项 ：创建一个可拖拽的引用项 ，可以将其拖放到检视面板中的引用抽屉中，以将所持有的对象分配给该变量（类似于 Unity 的拖放引用分配）。
Show Multi Selection Toggles: displays multi-selection toggles in front of each object. This is mostly useful on mobile devices where CTRL and Shift keys aren't present. Has no effect if Allow Multi Selection is disabled
显示多选切换按钮 ：在每个对象前面显示多选切换按钮。这在移动设备上尤其有用，因为移动设备上通常没有 Ctrl 和 Shift 键。如果 “允许多选” 已禁用，则此功能无效。
Show Multi Selection Toggles Then Create Dragged Reference Item: if multi-selection toggles aren't visible, displays them. Otherwise, creates a dragged reference item
显示多选切换按钮，然后创建拖动参考项 ：如果多选切换按钮不可见，则显示它们。否则，创建一个拖动参考项。
Pointer Long Press Duration: determines how long an object should be held until the Pointer Long Press Action is executed
指针长按持续时间 ：决定了在执行指针长按操作之前需要按住对象多长时间。
Double Click Threshold: when an object in the hierarchy is double clicked, OnItemDoubleClicked event is raised (see SCRIPTING API). This value determines the maximum allowed delay between two clicks to register a double click
双击阈值 ：当层级结构中的对象被双击时，会触发 OnItemDoubleClicked 事件（参见脚本 API ）。此值决定了两次点击之间允许的最大延迟，以判定是否触发双击事件。
Can Reorganize Items: when enabled, dropping a dragged reference item that holds Transform(s) onto an object in the hierarchy will change the dragged Transform(s)' parents (similar to parenting in Unity's Hierarchy)
可以重新组织项目 ：启用后，将包含变换的拖动引用项放到层级结构中的对象上，将更改拖动变换的父级（类似于 Unity 层级结构中的父级关系）。
Can Drop Dragged Parent On Child: when enabled, a dragged reference item can be dropped onto one of its child objects. In this case, the child object will be unparented and then the dragged reference item will become a child of it. Has no effect if Can Reorganize Items is disabled
可将拖动的父对象放置到子对象上 ：启用此功能后，可以将拖动的引用项放置到其子对象上。此时，子对象将失去父级关系，拖动的引用项将成为该子对象的子对象。如果 “可重新组织项目” 已禁用，则此功能无效。
Can Drop Dragged Objects To Pseudo Scenes: when enabled, dropping a dragged reference item onto a pseudo-scene or above/below a root object in the pseudo-scene will automatically add it to that pseudo-scene. Has no effect if Can Reorganize Items is disabled
可将拖动对象放置到伪场景中 ：启用此功能后，将拖动的参考项放置到伪场景中，或放置到伪场景中根对象的上方/下方，将自动将其添加到该伪场景中。如果 “可重新组织项目” 已禁用，则此功能无效。
Show Tooltips: when enabled, hovering over an object for a while will show a tooltip displaying the object's name. Can be useful for objects with very long names
显示工具提示 ：启用后，将鼠标悬停在对象上一段时间，将显示一个工具提示，其中包含对象的名称。这对于名称很长的对象非常有用。
Tooltip Delay: determines how long the cursor should remain static over an object before the tooltip appears. Has no effect if Show Tooltips is disabled
工具提示延迟 ：决定光标在对象上停留多长时间后才会显示工具提示。如果 “显示工具提示” 已禁用，则此设置无效。
Show Horizontal Scrollbar: when enabled, a horizontal scrollbar will be displayed if the names displayed in the hierarchy don't fit the available space. Note that only the visible items' width values are used to determine the size of the scrollable area
显示水平滚动条 ：启用后，如果层级结构中显示的名称超出可用空间，则会显示水平滚动条。请注意，滚动区域的大小仅根据可见项的宽度值来确定。
Sync Selection With Editor Hierarchy: simply synchronizes the selected object between the Unity Hierarchy and this RuntimeHierarchy
将选择与编辑器层级同步 ：简单地将 Unity 层级和此运行时层级之间的选定对象同步。
Additional settings for Can Reorganize Items can be found at the RuntimeHierarchy/ScrollView/Viewport object:
可以在 RuntimeHierarchy/ScrollView/Viewport 对象中找到 “可重新组织项目” 的其他设置：

screenshot

Sibling Index Modification Area: when a dragged reference item is dropped near the top or bottom edges of a Transform in hierarchy, it will be inserted above or belove the target Transform. This value determines the size of the area near the top and bottom edges
同级索引修改区域 ：当拖动的引用项放置在层级结构中变换的顶部或底部边缘附近时，它将被插入到目标变换的上方或下方。此值决定了顶部和底部边缘附近区域的大小。
Scrollable Area: while hovering the cursor near the top or bottom edges of the scroll view with a dragged reference item, scroll view will automatically be scrolled to show contents in that direction. This value determines the size of the area near the top and bottom edges of the scroll view
可滚动区域 ：当拖动参考项并将光标悬停在滚动视图的顶部或底部边缘附近时，滚动视图将自动滚动以显示该方向的内容。此值决定了滚动视图顶部和底部边缘附近区域的大小。
Scroll Speed: determines how fast the scroll view will be scrolled while hovering the cursor over Scrollable Area
滚动速度 ：决定当光标悬停在可滚动区域上时，滚动视图的滚动速度。
F. SCRIPTING API  F. 脚本 API
Values of the variables that are mentioned in E.1 and E.2 sections can be tweaked at runtime via their corresponding properties. Any changes to these properties will be reflected to UI immediately. Here, you will find some interesting things that you can do with the inspector and the hierarchy via scripting:
E.1 和 E.2 节中提到的变量值可以在运行时通过其对应的属性进行调整。对这些属性的任何更改都会立即反映在用户界面上。在这里，您会发现一些可以通过脚本使用检查器和层级结构实现的有趣功能：

You can change the inspected object in the inspector using the following functions:
您可以使用以下函数在检查器中更改被检查的对象：
public void Inspect( object obj );
public void StopInspect();
You can access the currently inspected object via the InspectedObject property of the inspector
您可以通过检查器的 InspectedObject 属性访问当前正在检查的对象。
You can change the selected object in the hierarchy using the following functions:
您可以使用以下功能更改层次结构中选定的对象：
// SelectOptions is an enum flag meaning that it can take multiple values with | (OR) operator. These values are:
// - Additive: new selection will be appended to the current selection instead of replacing it
// - FocusOnSelection: scroll view will be snapped to the selected object(s)
// - ForceRevealSelection: normally, when selection changes, the new selection will be fully explored in the hierarchy (i.e. all of the parents of the selection will be
//   expanded to reveal the selection). This doesn't automatically happen if selection doesn't change. When this flag is set, however, the selected objects will be fully
//   revealed/explored even if the selection doesn't change
public bool Select( Transform selection, SelectOptions selectOptions = SelectOptions.None ); // Selects the specified Transform. Returns true when the selection is changed successfully
public bool Select( IList<Transform> selection, SelectOptions selectOptions = SelectOptions.None ); // Selects the specified Transform(s)

public void Deselect(); // Deselects all Transforms
public void Deselect( Transform deselection ); // Deselects only the specified Transform
public void Deselect( IList<Transform> deselection ); // Deselects only the specified Transform(s)

public bool IsSelected( Transform transform ); // Returns true if the selection includes the Transform
You can access the currently selected object(s) in the hierarchy via the CurrentSelection property
您可以通过 CurrentSelection 属性访问层次结构中当前选定的对象。
Hierarchy's multi-selection toggles can be enabled manually via the MultiSelectionToggleSelectionMode property
可以通过 MultiSelectionToggleSelectionMode 属性手动启用层级结构的多选切换功能。
You can call the Refresh() function on the inspector and/or the hierarchy to refresh them manually
您可以手动调用检查器和/或层级视图上的 Refresh() 函数来刷新它们。
You can lock the inspector and/or the hierarchy via the IsLocked property
您可以通过 IsLocked 属性锁定检查器和/或层级结构。
You can register to the OnSelectionChanged event of the hierarchy to get notified when the selection has changed
您可以注册层级结构的 OnSelectionChanged 事件，以便在选择发生变化时收到通知。
You can register to the OnInspectedObjectChanging delegate of the inspector to get notified when the inspected object is about to change and, if you prefer, change the inspected object altogether. For example, if you want to inspect only objects that have a Renderer component attached, you can use the following function:
您可以注册到检查器的 OnInspectedObjectChanging 代理，以便在被检查对象即将更改时收到通知，如果您愿意，还可以直接更改被检查对象。例如，如果您只想检查附加了 Renderer 组件的对象，可以使用以下函数：
private object OnlyInspectObjectsWithRenderer( object previousInspectedObject, object newInspectedObject )
{
	GameObject go = newInspectedObject as GameObject;
	if( go != null && go.GetComponent<Renderer>() != null )
		return newInspectedObject;

	// Don't inspect objects without a Renderer component
	return null;
}
You can register to the ComponentFilter delegate of the inspector to filter the list of visible components of a GameObject in the inspector (e.g. hide some components)
您可以注册到检查器 ComponentFilter 委托，以过滤检查器中游戏对象的可见组件列表（例如，隐藏某些组件）。
runtimeInspector.ComponentFilter = ( GameObject gameObject, List<Component> components ) =>
{
    // Simply remove the undesired Components from the 'components' list
};
You can register to the GameObjectFilter delegate of the hierarchy to hide some objects from the hierarchy (or, you can add those objects to RuntimeInspectorUtils.IgnoredTransformsInHierarchy and they will be hidden from all hierarchies; just make sure to remove them from this HashSet before they are destroyed)
您可以向层级结构的 GameObjectFilter 代理注册，以隐藏层级结构中的某些对象（或者，您可以将这些对象添加到 RuntimeInspectorUtils.IgnoredTransformsInHierarchy 中，它们将从所有层级结构中隐藏；只需确保在销毁它们之前从该 HashSet 中移除它们）。
runtimeHierarchy.GameObjectFilter = ( Transform obj ) =>
{
    if( obj.CompareTag( "Main Camera" ) )
        return false; // Hide Main Camera from hierarchy
 
    return true;
};
You can register to the OnItemDoubleClicked event of the hierarchy to get notified when an object in the hierarchy is double clicked
您可以注册层级结构的 OnItemDoubleClicked 事件，以便在层级结构中的对象被双击时收到通知。
You can add RuntimeInspectorButton attribute to your functions to expose them as buttons in the inspector. These buttons appear when an object of that type is inspected. This attribute takes 3 parameters:
您可以为函数添加 RuntimeInspectorButton 特性，使其在检查器中显示为按钮。当检查该类型的对象时，这些按钮就会出现。此特性接受 3 个参数：
string label: the text that will appear on the button
字符串标签 ：按钮上将显示的文本
bool isInitializer: if set to true and the function returns an object that is assignable to the type that the function was defined in, the resulting value of the function will be assigned back to the inspected object. In other words, this function can be used to initialize null objects or change the variables of structs
bool isInitializer ：如果设置为 true，且函数返回的对象可以赋值给定义该函数的类型，则函数返回的值将被赋值给被检查的对象。换句话说，此函数可用于初始化空对象或更改结构体的变量。
ButtonVisibility visibility: determines when the button can be visible. Buttons with ButtonVisibility.InitializedObjects can appear only when the inspected object is not null whereas buttons with ButtonVisibility.UninitializedObjects can appear only when the inspected object is null. You can use ButtonVisibility.InitializedObjects | ButtonVisibility.UninitializedObjects to always show the button in the inspector
ButtonVisibility 属性决定按钮何时可见。值为 ButtonVisibility.InitializedObjects 的按钮仅在被检查对象不为空时显示，值为 ButtonVisibility.UninitializedObjects 的按钮仅在被检查对象为空时显示。您可以使用 ButtonVisibility.InitializedObjects | ButtonVisibility.UninitializedObjects 使按钮始终显示在检查器中。
Although you can't add RuntimeInspectorButton attribute to Unity's built-in functions, you can show buttons under built-in Unity types via extension methods. You must write all such extension methods in a single static class, mark the methods with RuntimeInspectorButton attribute and then introduce these functions to the RuntimeInspector as follows: RuntimeInspectorUtils.ExposedExtensionMethodsHolder = typeof( TheScriptThatContainsTheExtensionsMethods );
虽然不能将 RuntimeInspectorButton 特性添加到 Unity 的内置函数中，但可以通过扩展方法在 Unity 内置类型下显示按钮。您必须将所有此类扩展方法编写在一个静态类中，使用 RuntimeInspectorButton 特性标记这些方法，然后按如下方式将这些函数引入 RuntimeInspector： RuntimeInspectorUtils.ExposedExtensionMethodsHolder = typeof( TheScriptThatContainsTheExtensionsMethods );
F.1. PSEUDO-SCENES  F.1. 伪场景
You can use the following functions to add object(s) to pseudo-scenes in the hierarchy:
您可以使用以下函数向层级结构中的伪场景添加对象：

public void AddToPseudoScene( string scene, Transform transform );
public void AddToPseudoScene( string scene, IEnumerable<Transform> transforms );
These functions will create the relevant pseudo-scenes automatically if they do not exist.
如果相关的伪场景不存在，这些函数将自动创建它们。

You can use the following functions to remove object(s) from pseudo-scenes in the hierarchy:
您可以使用以下函数从层级结构中的伪场景中移除对象：

public void RemoveFromPseudoScene( string scene, Transform transform, bool deleteSceneIfEmpty );
public void RemoveFromPseudoScene( string scene, IEnumerable<Transform> transforms, bool deleteSceneIfEmpty );
You can use the following functions to create or delete a pseudo-scene manually:
您可以使用以下函数手动创建或删除伪场景：

public void CreatePseudoScene( string scene, Transform rootTransform = null );
public void DeletePseudoScene( string scene );
public void DeleteAllPseudoScenes();
The optional rootTransform parameter of CreatePseudoScene acts similar to PseudoSceneSourceTransform with the following differences:
CreatePseudoScene 的可选参数 rootTransform 与 PseudoSceneSourceTransform 的作用类似，但有以下区别：

Doesn't require adding a component to the source Transform
无需向源转换添加组件
When a Transform is dragged & dropped onto the pseudo-scene, its parent will actually be changed to rootTransform
当一个变换对象被拖放到伪场景中时，它的父对象实际上会更改为根变换对象。
During search, selected Transform's displayed path will stop at rootTransform (i.e. won't include its parents)
搜索过程中，所选变换的显示路径将止于根变换 （即不会包含其父变换）。
F.1.1. PseudoSceneSourceTransform
F.1.1. 伪场景源变换
This helper component allows you to add an object's children to a pseudo-scene in the hierarchy. When a child is added to or removed from the object, this component refreshes the pseudo-scene automatically. If HideOnDisable is enabled, the object's children are removed from the pseudo-scene when the object is disabled.
此辅助组件允许您将对象的子对象添加到层级结构中的伪场景。当子对象被添加或移除时，此组件会自动刷新伪场景。如果启用了 “禁用时隐藏” 功能，则当对象被禁用时，其子对象也会从伪场景中移除。

F.2. COLOR PICKER  F.2. 颜色选择器
You can access the built-in color picker via ColorPicker.Instance and then present it with the following function:
您可以通过 ColorPicker.Instance 访问内置颜色选择器，然后使用以下函数显示它：

public void Show( ColorWheelControl.OnColorChangedDelegate onColorChanged, ColorWheelControl.OnColorChangedDelegate onColorConfirmed, Color initialColor, Canvas referenceCanvas );
onColorChanged: invoked regularly as the user changes the color. ColorWheelControl.OnColorChangedDelegate takes a Color32 parameter
onColorChanged ：当用户更改颜色时定期调用。 ColorWheelControl.OnColorChangedDelegate 接受一个 Color32 参数。
onColorConfirmed: invoked when user submits the color via OK button
onColorConfirmed ：当用户通过 “确定” 按钮提交颜色时调用。
initialColor: the initial value of the color picker
initialColor ：颜色选择器的初始值
referenceCanvas: if assigned, the reference canvas' properties will be copied to the color picker canvas
referenceCanvas ：如果指定，则参考画布的属性将复制到颜色选择器画布。
You can change the color picker's visual appearance by assigning a UISkin to its Skin property.
您可以通过将 UISkin 分配给颜色选择器的 Skin 属性来更改其视觉外观。

F.3. OBJECT REFERENCE PICKER
F.3. 对象引用选择器
You can access the built-in object reference picker via ObjectReferencePicker.Instance and then present it with the following function:
您可以通过 ObjectReferencePicker.Instance 访问内置的对象引用选择器，然后使用以下函数显示它：

public void Show( ReferenceCallback onReferenceChanged, ReferenceCallback onSelectionConfirmed, NameGetter referenceNameGetter, NameGetter referenceDisplayNameGetter, object[] references, object initialReference, bool includeNullReference, string title, Canvas referenceCanvas );
onReferenceChanged: invoked when the user selects a reference from the list. ReferenceCallback takes an object parameter
onReferenceChanged ：当用户从列表中选择一个引用时调用。ReferenceCallback 接受一个对象参数 ReferenceCallback
onSelectionConfirmed: invoked when user submits the selected reference via OK button
onSelectionConfirmed ：当用户通过 “确定” 按钮提交所选参考文献时调用。
referenceNameGetter: NameGetter takes an object parameter and returns that object's name as string. The passed function will be used to sort the references list and compare the references' names with the search string
referenceNameGetter ： NameGetter 接受一个对象参数，并返回该对象的名称字符串。传入的函数将用于对引用列表进行排序，并将引用名称与搜索字符串进行比较。
referenceDisplayNameGetter: the passed function will be used to get display names for the references. Usually, the same function is passed to this parameter and the referenceNameGetter parameter
referenceDisplayNameGetter ：传入的函数将用于获取引用的显示名称。通常，此参数和 referenceNameGetter 参数会传入同一个函数。
references: array of references to pick from
参考文献 ：要从中选取的参考文献数组
initialReference: initially selected reference
initialReference ：初始选择的引用
includeNullReference: is set to true, a null reference option will be added to the top of the references list
includeNullReference ：如果设置为 true ，则会在引用列表顶部添加一个空引用选项。
title: title of the object reference picker
标题 ：对象引用选择器的标题
referenceCanvas: if assigned, the reference canvas' properties will be copied to the object reference picker canvas
referenceCanvas ：如果已赋值，则引用画布的属性将被复制到对象引用选择器画布。
You can change the object reference picker's visual appearance by assigning a UISkin to its Skin property.
您可以通过将 UISkin 分配给对象的 Skin 属性来更改对象引用选择器的视觉外观。

F.4. DRAGGED REFERENCE ITEMS
F.4. 拖拽的参考项目
In section E.2, it is mentioned that you can drag&drop objects from the hierarchy to the variables in the inspector to assign these objects to those variables. However, you are not limited with just hierarchy. There are two helper components that you can use to create dragged reference items for other objects:
在 E.2 节中提到，您可以将对象从层级结构拖放到检查器中的变量上，从而将这些对象分配给相应的变量。但是，您并不局限于层级结构。您可以使用两个辅助组件来创建其他对象的拖拽引用项：

DraggedReferenceSourceCamera: when attached to a camera, casts a ray to your scene at each mouse click and creates a dragged reference item if you hold on an object for a while. You can register to the ProcessRaycastHit delegate of this component to filter the objects than can create a dragged reference item. For example, if you want only objects with tag NPC to be able to create a dragged reference item, you can use the following function:
DraggedReferenceSourceCamera ：当附加到摄像机时，每次鼠标点击都会向场景投射一条射线，如果按住某个对象一段时间，则会创建一个拖动参考项。您可以注册此组件的 ProcessRaycastHit 代理来筛选可以创建拖动参考项的对象。例如，如果您只想让带有 NPC 标签的对象能够创建拖动参考项，可以使用以下函数：
private Object CreateDraggedReferenceItemForNPCsOnly( RaycastHit hit )
{
	if( hit.collider.gameObject.CompareTag( "NPC" ) )
		return hit.collider.gameObject;

	// Non-NPC objects can't create dragged reference items
	return null;
}
DraggedReferenceSourceUI: when assigned to a UI element, that element can create a dragged reference item for its References object(s) after it is clicked and held for a while
DraggedReferenceSourceUI ：当分配给一个 UI 元素时，该元素在被点击并按住一段时间后，可以为其 References 对象创建一个拖动引用项。
You can also use your own scripts to create dragged reference items by calling the following functions in the RuntimeInspectorUtils class:
您还可以通过调用 RuntimeInspectorUtils 类中的以下函数，使用自己的脚本创建拖动的参考项：

public static DraggedReferenceItem CreateDraggedReferenceItem( Object reference, PointerEventData draggingPointer, UISkin skin = null );
public static DraggedReferenceItem CreateDraggedReferenceItem( Object[] references, PointerEventData draggingPointer, UISkin skin = null, Canvas referenceCanvas = null );
G. CUSTOM DRAWERS (EDITORS)
G. 自定义抽屉（编辑）
NOTE: if you just want to hide some fields/properties from the RuntimeInspector, simply use Settings asset's Hidden Variables list (mentioned in section E.1).
注意： 如果您只想从 RuntimeInspector 中隐藏某些字段/属性，只需使用 “设置” 资源的 “隐藏变量” 列表（在 E.1 节中提到）。

You can introduce your own custom drawers to RuntimeInspector. These drawers will then be used to draw inspected objects' properties in RuntimeInspector. If no custom drawer is specified for a type, built-in ObjectField will be used to draw all properties of that type. There are 2 ways to create custom drawers:
您可以向 RuntimeInspector 添加自定义抽屉。这些抽屉将用于在 RuntimeInspector 中绘制被检查对象的属性。如果未为某个类型指定自定义抽屉，则将使用内置的 ObjectField 来绘制该类型的所有属性。创建自定义抽屉有两种方法：

Creating a drawer prefab and adding it to the Settings asset mentioned in section E.1. Each drawer extends from InspectorField base class. There is also an ExpandableInspectorField abstract class that allows you to create an expandable/collapsable drawer like arrays. Lastly, extending ObjectReferenceField class allows you to create drawers that can be assigned values via the reference picker or via drag&drop
创建抽屉预制件并将其添加到 E.1 节中提到的设置资源中。每个抽屉都继承自 InspectorField 基类。此外，还有一个 ExpandableInspectorField 抽象类，允许您创建类似数组的可展开/可折叠抽屉。最后，继承 ObjectReferenceField 类允许您创建可以通过引用选择器或拖放操作赋值的抽屉。
This option provides the most flexibility because you'll be able to customize the drawer prefab as you wish. The downside is, you'll have to create a prefab asset and manually add it to RuntimeInspector's Settings asset. All built-in drawers use this method; they can be as simple as BoolField and TransformField, or as complex as BoundsField, GameObjectField and ArrayField
此选项提供了最大的灵活性，因为您可以根据需要自定义抽屉预制件。缺点是，您必须创建一个预制件资源，并手动将其添加到 RuntimeInspector 的设置资源中。所有内置抽屉都使用此方法；它们可以像 BoolField 和 TransformField 一样简单，也可以像 BoundsField 、 GameObjectField 和 ArrayField 一样复杂。
Extending IRuntimeInspectorCustomEditor interface and decorating the class/struct with RuntimeInspectorCustomEditor attribute
扩展 IRuntimeInspectorCustomEditor 接口，并使用 RuntimeInspectorCustomEditor 特性修饰类/结构体。
This option is simpler because you won't have to create a prefab asset for the drawer. Created custom drawer will internally be used by ObjectField to populate its sub-drawers. This option should be sufficient for most use-cases. But imagine that you want to create a custom drawer for Matrix4x4 where the cells are displayed in a 4x4 grid. In this case, you must use the first option because you'll need a custom prefab with 16 InputFields organized in a 4x4 grid for it. But if you can represent the custom drawer you have in mind by using a combination of built-in drawers, then this second option should suffice
此选项更简单，因为您无需为抽屉创建预制件资源。ObjectField 内部会使用创建的自定义抽屉来填充其子抽屉。此选项足以满足大多数使用场景。但假设您想为 Matrix4x4 创建一个自定义抽屉，其中单元格以 4x4 网格形式显示。在这种情况下，您必须使用第一个选项，因为您需要一个包含 16 个以 4x4 网格排列的 InputField 的自定义预制件。但是，如果您可以通过组合使用内置抽屉来表示您设想的自定义抽屉，那么第二个选项就足够了。
G.1. InspectorField  G.1. 检查字段
To have a standardized visual appearance across all the drawers, there are some common variables for each drawer:
为了使所有抽屉的外观保持一致，每个抽屉都采用了一些通用变量：

Layout Element: is used to set the height of the drawer. A standard height is set by the currently active Inspector skin's Line Height property. This value is multiplied by the virtual HeightMultiplier property of the drawer. For ExpandableInspectorField's of unknown height, this variable should be left unassigned
布局元素 ：用于设置抽屉的高度。标准高度由当前激活的检查器皮肤的 “行高” 属性设置。该值乘以抽屉的虚拟 “高度乘数” 属性。对于高度未知的 ExpandableInspectorField，此变量应保持未赋值状态。
Variable Name Text: the Text object that displays the name of the exposed variable
变量名称文本 ：显示已公开变量名称的文本对象
Variable Name Mask: to understand this one, you may have to examine a simple drawer like BoolField. An Image is drawn on top of the Variable Name Text in order to mask its visible area in an efficient way. And this mask is assigned to this variable
变量名掩码 ：要理解这一点，您可能需要查看一个简单的绘图器，例如 BoolField。在变量名文本上方绘制一个图像 ，以有效地遮盖其可见区域。并且此掩码被分配给该变量。
Each drawer has access to the following properties:
每个抽屉都可以访问以下属性：

object Value: the most recent value of the variable that this drawer is bound to. It is refreshed at each refresh interval of the inspector. Changing this property will also change the bound object
对象值 ：此抽屉绑定的变量的最新值。它会在检查器每次刷新间隔时刷新。更改此属性也会更改绑定的对象。
RuntimeInspector Inspector: the RuntimeInspector that currently uses this drawer
RuntimeInspector 检查器 ：当前使用此抽屉的 RuntimeInspector
UISkin Skin: the skin that is assigned to this drawer
UISkin 皮肤 ：分配给此抽屉的皮肤
Type BoundVariableType: the type of the bound object
BoundVariableType 类型 ：绑定对象的类型
int Depth: the depth that this drawer is drawn at. As Depth increases, a padding should be applied to the contents of this drawer from left (in OnDepthChanged function)
int Depth ：此抽屉的绘制深度。随着 Depth 的增加，应在此抽屉的内容左侧添加内边距（在 OnDepthChanged 函数中）。
string Name: the name of the bound variable. When set, the variable name is converted to title case format if Use Title Case Naming is enabled in the inspector
字符串名称 ：绑定变量的名称。如果在检查器中启用了 “使用首字母大写命名” ，则设置此参数后，变量名称将转换为首字母大写格式。
string NameRaw: When set, the variable name is used as is without being converted to title case format
string NameRaw ：设置后，变量名将按原样使用，而不转换为首字母大写格式。
float HeightMultiplier: affects the height of the drawer
float HeightMultiplier ：影响抽屉的高度
There are some special functions on drawers that are invoked on certain circumstances:
抽屉程序有一些特殊功能，会在特定情况下调用：

void Initialize(): should be used instead of Awake/Start to initialize the drawer
void Initialize() ：应该使用该方法代替 Awake / Start 来初始化抽屉。
bool SupportsType( Type type ): returns whether or not this drawer can expose (supports) a certain type in the inspector
bool SupportsType(类型 type) : 返回此抽屉是否可以在检查器中显示（支持）特定类型
bool CanBindTo( Type type, MemberInfo variable ): returns whether or not this drawer can expose the provided variable. This function is called only if SupportsType returns true. This function is useful for drawers that can expose only variables with specific attribute(s) (e.g. NumberRangeField queries RangeAttribute). Please note that the variable parameter can be null. By default, this function returns true
bool CanBindTo(Type type, MemberInfo variable) : 返回此抽屉是否可以公开提供的变量 。仅当 SupportsType 返回 true 时才会调用此函数。此函数适用于只能公开具有特定属性的变量的抽屉（例如， NumberRangeField 查询 RangeAttribute）。请注意， variable 参数可以为 null 。默认情况下，此函数返回 true。
void OnBound( MemberInfo variable ): called when the drawer is bound to a variable via reflection. Please note that the variable parameter can be null
void OnBound( MemberInfo variable ) ：当抽屉通过反射绑定到变量时调用。请注意， 变量参数可以为空。
void OnUnbound(): called when the drawer is unbound from the variable that it was bound to
void OnUnbound() ：当抽屉从其绑定的变量中解除绑定时调用
void OnInspectorChanged(): called when the Inspector property of the drawer is changed
void OnInspectorChanged() ：当抽屉的 Inspector 属性更改时调用
void OnSkinChanged(): called when the Skin property of the drawer is changed. Your custom drawers must adjust their UI elements' visual appearance here to comply with the assigned skin's standards
void OnSkinChanged() ：当抽屉的 Skin 属性发生更改时调用。您的自定义抽屉必须在此处调整其 UI 元素的视觉外观，以符合所分配皮肤的标准。
void OnDepthChanged(): called when the Depth property of the drawer is changed. Here, your custom drawers must add a padding to their content from left to comply with the nesting standard. This function is also called when the Skin changes
void OnDepthChanged() ：当抽屉的 Depth 属性发生变化时调用。此时，您的自定义抽屉必须为其内容添加左侧内边距，以符合嵌套标准。当皮肤发生变化时，也会调用此函数。
void Refresh(): called when the value of the bound object is refreshed. Drawers must refresh the values of their UI elements here. Invoked by RuntimeInspector at every Refresh Interval seconds
void Refresh() ：当绑定对象的值刷新时调用。抽屉必须在此处刷新其 UI 元素的值。RuntimeInspector 每隔 Refresh Interval 秒调用一次。
G.2. ExpandableInspectorField
G.2. 可扩展检查器字段
Custom drawers that extend ExpandableInspectorField have access to the following properties:
继承自 ExpandableInspectorField 的自定义抽屉可以访问以下属性：

bool IsExpanded: returns whether the drawer is expanded or collapsed. When set to true, the drawer is expanded and its contents are drawn under it
bool IsExpanded ：返回抽屉是展开还是折叠状态。设置为 true 时，抽屉展开，其内容显示在抽屉下方。
HeaderVisibility HeaderVisibility: sets the visibility of this drawer's header: Collapsible, AlwaysVisible or Hidden. By default, this value is set to Collapsible
HeaderVisibility ：设置此抽屉标题的可见性： 可折叠 、 始终可见或隐藏 。默认值为可折叠。
int Length: the number of elements that this drawer aims to draw. If its value does not match the number of child drawers that this drawer has, the contents of the drawer are regenerated
int Length ：此抽屉要绘制的元素数量。如果其值与此抽屉拥有的子抽屉数量不匹配，则重新生成抽屉的内容。
ExpandableInspectorField has the following special functions:
ExpandableInspectorField 具有以下特殊功能：

void GenerateElements(): the sub-drawers of this drawer must be generated here
void GenerateElements() ：必须在此处生成此抽屉的子抽屉。
void ClearElements(): the sub-drawers of this drawer must be cleared here
void ClearElements() ：必须在此处清除此抽屉的子抽屉。
Sub-drawers of an ExpandableInspectorField should be stored in the protected List<InspectorField> elements variable as ExpandableInspectorField uses this list to compare the number of sub-drawers with the Length property. When Refresh() is called, sub-drawers in this list are refreshed automatically and when ClearElements() is called, sub-drawers in this list are cleared automatically.
ExpandableInspectorField 的子抽屉应存储在 protected List<InspectorField> elements 变量中，因为 ExpandableInspectorField 使用此列表将子抽屉的数量与 Length 属性进行比较。 调用 Refresh() 时，此列表中的子抽屉会自动刷新；调用 ClearElements() 时，此列表中的子抽屉会自动清除。

You can create sub-drawers using the RuntimeInspector.CreateDrawerForType( Type type, Transform drawerParent, int depth, bool drawObjectsAsFields = true ) function. If no drawer is found that can expose this type, the function returns null. Here, for ExpandableInspectorFields, the drawerParent parameter should be set as the drawArea variable of the ExpandableInspectorField. If the drawObjectsAsFields parameter is set to true and if the type extends UnityEngine.Object, Reference Drawers are searched for a drawer that supports this type. Otherwise Standard Drawers are searched.
您可以使用 RuntimeInspector.CreateDrawerForType( Type type, Transform drawerParent, int depth, bool drawObjectsAsFields = true ) 函数创建子绘制器。如果找不到可以公开此类型的绘制器，该函数将返回 null 。对于 ExpandableInspectorFields， drawerParent 参数应设置为 ExpandableInspectorField 的 drawArea 变量。如果 drawObjectsAsFields 参数设置为 true，并且该类型继承自 UnityEngine.Object ，则会在引用绘制器中搜索支持此类型的绘制器。否则，将搜索标准绘制器 。

After creating sub-drawers, ExpandableInspectorFields must bind their sub-drawers to their corresponding variables manually. This is done via the following BindTo functions of the InspectorField class:
创建子抽屉后， ExpandableInspectorField 必须手动将其子抽屉绑定到相应的变量。这可以通过 InspectorField 类的以下 BindTo 函数完成：

BindTo( InspectorField parent, MemberInfo variable, string variableName = null ): binds the object to a MemberInfo (it can be received via reflection). Here, parent parameter should be set to this ExpandableInspectorField. If variableName is set to null, its value is fetched directly from the MemberInfo parameter
BindTo( InspectorField parent, MemberInfo variable, string variableName = null ) ：将对象绑定到 MemberInfo （可通过反射获取）。此处， parent 参数应设置为此 ExpandableInspectorField 。如果 variableName 设置为 null，则其值直接从 MemberInfo 参数中获取。
BindTo( Type variableType, string variableName, Getter getter, Setter setter, MemberInfo variable = null ): this one allows you to define your own getter and setter functions for this sub-drawer. For example, ArrayField uses this function because there is no direct MemberInfo to access an element of an array. With this method, you can use custom functions instead of MemberInfos to get/set the values of the bound objects (ArrayField uses Array.GetValue for its elements' getter function and Array.SetValue for its elements' setter function)
BindTo( Type variableType, string variableName, Getter getter, Setter setter, MemberInfo variable = null ) ：此选项允许您为该子抽屉定义自定义的 getter 和 setter 函数。例如， ArrayField 使用此函数，因为没有直接的 MemberInfo 可以访问数组元素。通过此方法，您可以使用自定义函数而不是 MemberInfo 来获取/设置绑定对象的值（ArrayField 使用 Array.GetValue 作为其元素的 getter 函数，使用 Array.SetValue 作为其元素的 setter 函数）。
There are also some helper functions in ExpandableInspectorField to easily create sub-drawers without having to call CreateDrawerForType or BindTo manually:
ExpandableInspectorField 中还有一些辅助函数，可以轻松创建子抽屉，而无需手动调用 CreateDrawerForType 或 BindTo ：

InspectorField CreateDrawerForComponent( Component component, string variableName = null ): creates a Standard Drawer for a component
InspectorField CreateDrawerForComponent( Component component, string variableName = null ) ：为组件创建一个标准抽屉。
InspectorField CreateDrawerForVariable( MemberInfo variable, string variableName = null ): creates a drawer for the variable that the MemberInfo stores. This variable must be declared inside inspected object's class/struct or one of its base classes
InspectorField CreateDrawerForVariable( MemberInfo variable, string variableName = null ) ：为 MemberInfo 存储的变量创建一个抽屉。此变量必须在被检查对象的类/结构体或其基类之一中声明。
InspectorField CreateDrawer( Type variableType, string variableName, Getter getter, Setter setter, bool drawObjectsAsFields = true ): similar to the BindTo function with the Getter and Setter parameters, allows you to use custom functions to get and set the value of the object that the sub-drawer is bound to
InspectorField CreateDrawer( Type variableType, string variableName, Getter getter, Setter setter, bool drawObjectsAsFields = true ) ：类似于带有 Getter 和 Setter 参数的 BindTo 函数，允许您使用自定义函数来获取和设置子抽屉绑定对象的值。
G.3. ObjectReferenceField
Drawers that extend ObjectReferenceField class have access to the void OnReferenceChanged( Object reference ) function that is called when the reference assigned to that drawer is changed.

G.4. Helper Classes
PointerEventListener: this is a simple helper component that invokes PointerDown event when its UI GameObject is pressed, PointerUp event when it is released and PointerClick event when it is clicked

BoundInputField: most of the built-in drawers use this component for their input fields. This helper component allows you to validate the input as it is entered and also get notified when the input is submitted. It has the following properties and functions:

string DefaultEmptyValue: the default value that the input field will have when its input is empty. For example, NumberField sets this value to "0"
string Text: a property to refresh the current value of the input field. If the input field is currently focused and being edited, then this property will not change its text immediately but store the value in a variable so that it can be used when the input field is no longer focused. Also, setting this property will not invoke the OnValueChanged event
UISkin Skin: the skin that this input field uses. When set, input field will adjust its UI accordingly
OnValueChangedDelegate OnValueChanged: called while the value of input field is being edited (called at each change to the input). The OnValueChangedDelegate has the following signature: bool OnValueChangedDelegate( BoundInputField source, string input ). A function that is registered to this event should parse the input and return true if the input is valid, false otherwise
OnValueChangedDelegate OnValueSubmitted: called when user finishes editing the value of input field. Similar to OnValueChanged, a function that is registered to this event should parse the input and return true only if the input is valid
bool CacheTextOnValueChange: determines what will happen when user stops editing the input field while its contents are invalid (i.e. its background has turned red). If this variable is set to true, input field's text will revert to the latest value that returned true for OnValueChanged. Otherwise, the text will revert to the value input field had when it was focused
G.5. RuntimeInspectorCustomEditor Attribute
To create drawers without having to create a prefab for it, you can declara a class/struct that extends IRuntimeInspectorCustomEditor and has one or more RuntimeInspectorCustomEditor attributes.

RuntimeInspectorCustomEditor attribute has the following properties:

Type inspectedType: the type this custom drawer supports (can expose)
bool editorForChildClasses: if set to true, types derived from inspectedType can also be drawn with this drawer. By default, this value is false
IRuntimeInspectorCustomEditor has the following functions:

void GenerateElements( ObjectField parent ): called by built-in ObjectField's GenerateElements function. Sub-drawers should be added to ObjectField in this function
void Refresh(): called by ObjectField's Refresh function
void Cleanup(): called by ObjectField's ClearElements function. If the drawer has created some disposable resources, they must be disposed here. No need to destroy the created sub-drawers here because it is handled by ObjectField automatically, as explained in ExpandableInspectorField section
Inside GenerateElements function, you can call parent parameter's CreateDrawerForComponent, CreateDrawerForVariable and CreateDrawer functions to create sub-drawers. In addition to these, you can also call the following helper functions of ObjectField:

void CreateDrawersForVariables( params string[] variables ): creates drawers for the specified variables of the inspected object. If no specific variables are provided, drawers will be created for all exposed variables of the inspected object
void CreateDrawersForVariablesExcluding( params string[] variablesToExclude ): creates drawers for all exposed variables of the inspected object except the variables specified in variablesToExclude list. If no variables are excluded, drawers will be created for all exposed variables of the inspected object
Here are some example custom drawers:

screenshot

// Custom drawer for Collider type and the types that derive from it
[RuntimeInspectorCustomEditor( typeof( Collider ), true )]
public class ColliderEditor : IRuntimeInspectorCustomEditor
{
	public void GenerateElements( ObjectField parent )
	{
		// Exposes only "enabled" and "isTrigger" properties of Colliders
		// Note that we could achieve the same thing by modifying the "Hidden Variables" and "Exposed Variables" lists of RuntimeInspector's Settings asset
		parent.CreateDrawersForVariables( "enabled", "isTrigger" );
	}

	public void Refresh() { }
	public void Cleanup() { }
}
screenshot

// Custom drawer for MeshRenderer type (but not the types that derive from it)
[RuntimeInspectorCustomEditor( typeof( MeshRenderer ), false )]
public class MeshRendererEditor : IRuntimeInspectorCustomEditor
{
	public void GenerateElements( ObjectField parent )
	{
		// Get the MeshRenderer object we are inspecting
		MeshRenderer renderer = (MeshRenderer) parent.Value;

		// Instead of exposing the MeshRenderer's properties, expose its sharedMaterial's properties
		ExpandableInspectorField materialField = (ExpandableInspectorField) parent.CreateDrawer( typeof( Material ), "", () => renderer.sharedMaterial, ( value ) => renderer.sharedMaterial = (Material) value, false );

		// The drawer for materials is, by default, an ExpandableInspectorField. We don't want to draw its collapsible header in this example
		materialField.HeaderVisibility = RuntimeInspector.HeaderVisibility.Hidden;
	}

	public void Refresh() { }
	public void Cleanup() { }
}
screenshot

// Custom drawer for Camera type (but not the types that derive from it)
[RuntimeInspectorCustomEditor( typeof( Camera ), false )]
public class CameraEditor : IRuntimeInspectorCustomEditor
{
	// Some of the sub-drawers that are created inside GenerateElements
	private BoolField isOrthographicField;
	private NumberField orthographicSizeField, fieldOfViewField;

	public void GenerateElements( ObjectField parent )
	{
		// Create sub-drawers for the Camera's "orthographic", "orthographicSize" and "fieldOfView" properties and store them in variables
		isOrthographicField = (BoolField) parent.CreateDrawerForVariable( typeof( Camera ).GetProperty( "orthographic", BindingFlags.Public | BindingFlags.Instance ), "Is Orthographic" );
		orthographicSizeField = (NumberField) parent.CreateDrawerForVariable( typeof( Camera ).GetProperty( "orthographicSize", BindingFlags.Public | BindingFlags.Instance ) );
		fieldOfViewField = (NumberField) parent.CreateDrawerForVariable( typeof( Camera ).GetProperty( "fieldOfView", BindingFlags.Public | BindingFlags.Instance ) );

		// Add additional indentation for "orthographicSize" and "fieldOfView" sub-drawers
		orthographicSizeField.Depth++;
		fieldOfViewField.Depth++;

		// Create sub-drawers for the rest of the exposed properties of the Camera
		parent.CreateDrawersForVariablesExcluding( "orthographic", "orthographicSize", "fieldOfView" );
	}

	public void Refresh()
	{
		// Check if Camera is currently using orthographic projection
		bool isOrthographicCamera = (bool) isOrthographicField.Value;

		// Show either "orthographicSize" sub-drawer or "fieldOfView" sub-drawer depending on camera's current projection type
		// (Here, we're first checking if the sub-drawer is already active/inactive via 'activeSelf' for optimization purposes because GameObject.SetActive
		// causes considerable GC allocations and unfortunately doesn't automatically check if GameObject is already active/inactive, at least on some Unity versions)
		if( orthographicSizeField.gameObject.activeSelf != isOrthographicCamera )
			orthographicSizeField.gameObject.SetActive( isOrthographicCamera );
		if( fieldOfViewField.gameObject.activeSelf == isOrthographicCamera )
			fieldOfViewField.gameObject.SetActive( !isOrthographicCamera );
	}

	public void Cleanup() { }
}
About
Runtime Inspector and Hierarchy solution for Unity for debugging and runtime editing purposes

Resources
 Readme
License
 MIT license
Contributing
 Contributing
 Activity
Stars
 2.1k stars
Watchers
 41 watching
Forks
 148 forks
Report repository
Releases 20
v1.7.4
Latest
on Mar 29, 2025
+ 19 releases
Sponsor this project
@yasirkula
yasirkula
https://yasirkula.itch.io/unity3d/donate
Learn more about GitHub Sponsors
Packages
No packages published
Contributors
2
@yasirkula
yasirkula
@i-xt
i-xt
Languages
C#
98.1%
 
ShaderLab
1.9%
Footer
© 2026 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Community
Docs
Contact
Manage cookies
Do not share my personal information
