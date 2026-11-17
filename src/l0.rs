#![allow(nonstandard_style)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::ffi::{c_char, c_void};

pub type ze_result_t = i32;
pub const ZE_RESULT_SUCCESS: ze_result_t = 0;

pub type ze_driver_handle_t = *mut c_void;
pub type ze_device_handle_t = *mut c_void;
pub type ze_context_handle_t = *mut c_void;
pub type ze_command_queue_handle_t = *mut c_void;
pub type ze_command_list_handle_t = *mut c_void;
pub type ze_module_handle_t = *mut c_void;
pub type ze_kernel_handle_t = *mut c_void;

pub type ze_init_driver_type_flags_t = u32;
pub const ZE_INIT_DRIVER_TYPE_FLAG_GPU: ze_init_driver_type_flags_t = 1 << 0;
pub const ZE_INIT_DRIVER_TYPE_FLAG_NPU: ze_init_driver_type_flags_t = 1 << 1;

pub const ZE_STRUCTURE_TYPE_INIT_DRIVER_DESC: i32 = 0x1;

#[repr(C)]
pub struct ze_init_driver_desc_t {
    pub stype: i32,
    pub pNext: *const c_void,
    pub flags: ze_init_driver_type_flags_t,
}

pub const ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES: i32 = 0x4;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ze_device_properties_t {
    pub stype: i32,
    pub pNext: *const c_void,
    pub type_: u32,
    pub vendorId: u32,
    pub deviceId: u32,
    pub flags: u32,
    pub subdeviceId: u32,
    pub coreClockRate: u32,
    pub maxCommandQueuePriority: u32,
    pub numAsyncComputeEngines: u32,
    pub numAsyncCopyEngines: u32,
    pub maxHardwareContexts: u32,
    pub name: [c_char; 256],
    pub uuid: [u8; 16],
}

pub type ze_command_queue_group_property_flags_t = u32;
pub const ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE: ze_command_queue_group_property_flags_t =
    1 << 0;
pub const ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY: ze_command_queue_group_property_flags_t =
    1 << 1;
pub const ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_METRICS: ze_command_queue_group_property_flags_t =
    1 << 2;
pub const ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS:
    ze_command_queue_group_property_flags_t = 1 << 3;

pub const ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES: i32 = 0x5;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ze_command_queue_group_properties_t {
    pub stype: i32,
    pub pNext: *const c_void,
    pub flags: ze_command_queue_group_property_flags_t,
    pub maxMemoryFillPatternSize: usize,
    pub numQueues: u32,
}

pub type ze_context_flags_t = u32;
pub const ZE_CONTEXT_FLAG_NONE: ze_context_flags_t = 0;
pub const ZE_CONTEXT_FLAG_TDR_DISABLE: ze_context_flags_t = 1 << 0;

pub const ZE_STRUCTURE_TYPE_CONTEXT_DESC: i32 = 0x2;

#[repr(C)]
pub struct ze_context_desc_t {
    pub stype: i32,
    pub pNext: *const c_void,
    pub flags: ze_context_flags_t,
}

pub type ze_command_queue_flags_t = u32;
pub const ZE_COMMAND_QUEUE_FLAG_NONE: ze_command_queue_flags_t = 0;
pub const ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY: ze_command_queue_flags_t = 1 << 0;
pub const ZE_COMMAND_QUEUE_FLAG_IN_ORDER: ze_command_queue_flags_t = 1 << 1;
pub const ZE_COMMAND_QUEUE_FLAG_COPY_OFFLOAD_HINT: ze_command_queue_flags_t = 1 << 2;

pub type ze_command_queue_mode_t = i32;
pub const ZE_COMMAND_QUEUE_MODE_DEFAULT: ze_command_queue_mode_t = 0;
pub const ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS: ze_command_queue_mode_t = 1;
pub const ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS: ze_command_queue_mode_t = 2;

pub type ze_command_queue_priority_t = i32;
pub const ZE_COMMAND_QUEUE_PRIORITY_NORMAL: ze_command_queue_priority_t = 0;
pub const ZE_COMMAND_QUEUE_PRIORITY_LOW: ze_command_queue_priority_t = 1;
pub const ZE_COMMAND_QUEUE_PRIORITY_HIGH: ze_command_queue_priority_t = 2;

pub const ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC: i32 = 0x3;

#[repr(C)]
pub struct ze_command_queue_desc_t {
    pub stype: i32,
    pub pNext: *const c_void,
    pub ordinal: u32,
    pub index: u32,
    pub flags: ze_command_queue_flags_t,
    pub mode: ze_command_queue_mode_t,
    pub priority: ze_command_queue_priority_t,
}

pub type ze_command_list_flags_t = u32;
pub const ZE_COMMAND_LIST_FLAG_NONE: ze_command_list_flags_t = 0;
pub const ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING: ze_command_list_flags_t = 1 << 0;
pub const ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT: ze_command_list_flags_t = 1 << 1;
pub const ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY: ze_command_list_flags_t = 1 << 2;

pub const ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC: i32 = 0x1;

#[repr(C)]
pub struct ze_command_list_desc_t {
    pub stype: i32,
    pub pNext: *const c_void,
    pub commandQueueGroupOrdinal: u32,
    pub flags: ze_command_list_flags_t,
}

pub type ze_device_mem_alloc_flags_t = u32;
pub const ZE_DEVICE_MEM_ALLOC_FLAG_NONE: ze_device_mem_alloc_flags_t = 0;
pub const ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED: ze_device_mem_alloc_flags_t = 1 << 0;
pub const ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED: ze_device_mem_alloc_flags_t = 1 << 1;

pub const ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC: i32 = 0x6;

#[repr(C)]
pub struct ze_device_mem_alloc_desc_t {
    pub stype: i32,
    pub pNext: *const c_void,
    pub flags: ze_device_mem_alloc_flags_t,
    pub ordinal: u32,
}

pub type ze_host_mem_alloc_flags_t = u32;
pub const ZE_HOST_MEM_ALLOC_FLAG_NONE: ze_host_mem_alloc_flags_t = 0;
pub const ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED: ze_host_mem_alloc_flags_t = 1 << 0;
pub const ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED: ze_host_mem_alloc_flags_t = 1 << 1;
pub const ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINE: ze_host_mem_alloc_flags_t = 1 << 2;

pub const ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC: i32 = 0x7;

#[repr(C)]
pub struct ze_host_mem_alloc_desc_t {
    pub stype: i32,
    pub pNext: *const c_void,
    pub flags: ze_host_mem_alloc_flags_t,
}

pub type ze_module_format_t = i32;
pub const ZE_MODULE_FORMAT_SPIRV: ze_module_format_t = 0;
pub const ZE_MODULE_FORMAT_NATIVE: ze_module_format_t = 1;

pub const ZE_STRUCTURE_TYPE_MODULE_DESC: i32 = 0x9;

#[repr(C)]
pub struct ze_module_desc_t {
    pub stype: i32,
    pub pNext: *const c_void,
    pub format: ze_module_format_t,
    pub inputSize: usize,
    pub pInputModule: *const u8,
    pub pBuildFlags: *const c_char,
    pub pConstants: *const c_void,
}

pub type ze_kernel_flags_t = u32;
pub const ZE_KERNEL_FLAG_NONE: ze_kernel_flags_t = 0;

pub const ZE_STRUCTURE_TYPE_KERNEL_DESC: i32 = 0xa;

#[repr(C)]
pub struct ze_kernel_desc_t {
    pub stype: i32,
    pub pNext: *const c_void,
    pub flags: ze_kernel_flags_t,
    pub pKernelName: *const c_char,
}

#[repr(C)]
pub struct ze_group_count_t {
    pub groupCountX: u32,
    pub groupCountY: u32,
    pub groupCountZ: u32,
}

#[link(name = "ze_loader")]
unsafe extern "C" {
    pub fn zeInitDrivers(
        pCount: *mut u32,
        phDrivers: *mut ze_driver_handle_t,
        desc: *const ze_init_driver_desc_t,
    ) -> ze_result_t;
    pub fn zeDeviceGet(
        hDriver: ze_driver_handle_t,
        pCount: *mut u32,
        phDevices: *mut ze_device_handle_t,
    ) -> ze_result_t;
    pub fn zeDeviceGetProperties(
        hDevice: ze_device_handle_t,
        pProperties: *mut ze_device_properties_t,
    ) -> ze_result_t;
    pub fn zeDeviceGetCommandQueueGroupProperties(
        hDevice: ze_device_handle_t,
        pCount: *mut u32,
        pProperties: *mut ze_command_queue_group_properties_t,
    ) -> ze_result_t;

    pub fn zeContextCreate(
        hDriver: ze_driver_handle_t,
        desc: *const ze_context_desc_t,
        phContext: *mut ze_context_handle_t,
    ) -> ze_result_t;
    pub fn zeContextDestroy(hContext: ze_context_handle_t) -> ze_result_t;

    pub fn zeCommandQueueCreate(
        hContext: ze_context_handle_t,
        hDevice: ze_device_handle_t,
        desc: *const ze_command_queue_desc_t,
        phCommandQueue: *mut ze_command_queue_handle_t,
    ) -> ze_result_t;
    pub fn zeCommandQueueDestroy(hCommandQueue: ze_command_queue_handle_t) -> ze_result_t;
    pub fn zeCommandQueueExecuteCommandLists(
        hCommandQueue: ze_command_queue_handle_t,
        numCommandLists: u32,
        phCommandLists: *const ze_command_list_handle_t,
        hFence: *mut c_void,
    ) -> ze_result_t;
    pub fn zeCommandQueueSynchronize(
        hCommandQueue: ze_command_queue_handle_t,
        timeout: u64,
    ) -> ze_result_t;

    pub fn zeCommandListCreate(
        hContext: ze_context_handle_t,
        hDevice: ze_device_handle_t,
        desc: *const ze_command_list_desc_t,
        phCommandList: *mut ze_command_list_handle_t,
    ) -> ze_result_t;
    pub fn zeCommandListClose(hCommandList: ze_command_list_handle_t) -> ze_result_t;
    pub fn zeCommandListDestroy(hCommandList: ze_command_list_handle_t) -> ze_result_t;
    pub fn zeMemAllocShared(
        hContext: ze_context_handle_t,
        device_desc: *const ze_device_mem_alloc_desc_t,
        host_desc: *const ze_host_mem_alloc_desc_t,
        size: usize,
        alignment: usize,
        hDevice: ze_device_handle_t,
        pptr: *mut *mut c_void,
    ) -> ze_result_t;
    pub fn zeMemFree(hContext: ze_context_handle_t, ptr: *mut c_void) -> ze_result_t;
    pub fn zeModuleCreate(
        hContext: ze_context_handle_t,
        hDevice: ze_device_handle_t,
        desc: *const ze_module_desc_t,
        phModule: *mut ze_module_handle_t,
        phBuildLog: *mut c_void,
    ) -> ze_result_t;
    pub fn zeModuleDestroy(hModule: ze_module_handle_t) -> ze_result_t;
    pub fn zeKernelCreate(
        hModule: ze_module_handle_t,
        desc: *const ze_kernel_desc_t,
        phKernel: *mut ze_kernel_handle_t,
    ) -> ze_result_t;
    pub fn zeKernelDestroy(hKernel: ze_kernel_handle_t) -> ze_result_t;
    pub fn zeKernelSetArgumentValue(
        hKernel: ze_kernel_handle_t,
        argIndex: u32,
        argSize: usize,
        pArgValue: *const c_void,
    ) -> ze_result_t;
    pub fn zeKernelSetGroupSize(
        hKernel: ze_kernel_handle_t,
        groupSizeX: u32,
        groupSizeY: u32,
        groupSizeZ: u32,
    ) -> ze_result_t;
    pub fn zeCommandListAppendLaunchKernel(
        hCommandList: ze_command_list_handle_t,
        hKernel: ze_kernel_handle_t,
        pLaunchFuncArgs: *const ze_group_count_t,
        hSignalEvent: *mut c_void,
        numWaitEvents: u32,
        phWaitEvents: *mut *mut c_void,
    ) -> ze_result_t;
}
