// Generated by gencpp from file xpkg_comm/xmsg_device.msg
// DO NOT EDIT!


#ifndef XPKG_COMM_MESSAGE_XMSG_DEVICE_H
#define XPKG_COMM_MESSAGE_XMSG_DEVICE_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace xpkg_comm
{
template <class ContainerAllocator>
struct xmsg_device_
{
  typedef xmsg_device_<ContainerAllocator> Type;

  xmsg_device_()
    : dev_class(0)
    , dev_type(0)
    , dev_number(0)
    , dev_enable(0)  {
    }
  xmsg_device_(const ContainerAllocator& _alloc)
    : dev_class(0)
    , dev_type(0)
    , dev_number(0)
    , dev_enable(0)  {
  (void)_alloc;
    }



   typedef uint8_t _dev_class_type;
  _dev_class_type dev_class;

   typedef uint8_t _dev_type_type;
  _dev_type_type dev_type;

   typedef uint8_t _dev_number_type;
  _dev_number_type dev_number;

   typedef uint8_t _dev_enable_type;
  _dev_enable_type dev_enable;





  typedef boost::shared_ptr< ::xpkg_comm::xmsg_device_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::xpkg_comm::xmsg_device_<ContainerAllocator> const> ConstPtr;

}; // struct xmsg_device_

typedef ::xpkg_comm::xmsg_device_<std::allocator<void> > xmsg_device;

typedef boost::shared_ptr< ::xpkg_comm::xmsg_device > xmsg_devicePtr;
typedef boost::shared_ptr< ::xpkg_comm::xmsg_device const> xmsg_deviceConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::xpkg_comm::xmsg_device_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::xpkg_comm::xmsg_device_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::xpkg_comm::xmsg_device_<ContainerAllocator1> & lhs, const ::xpkg_comm::xmsg_device_<ContainerAllocator2> & rhs)
{
  return lhs.dev_class == rhs.dev_class &&
    lhs.dev_type == rhs.dev_type &&
    lhs.dev_number == rhs.dev_number &&
    lhs.dev_enable == rhs.dev_enable;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::xpkg_comm::xmsg_device_<ContainerAllocator1> & lhs, const ::xpkg_comm::xmsg_device_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace xpkg_comm

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::xpkg_comm::xmsg_device_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::xpkg_comm::xmsg_device_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::xpkg_comm::xmsg_device_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::xpkg_comm::xmsg_device_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::xpkg_comm::xmsg_device_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::xpkg_comm::xmsg_device_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::xpkg_comm::xmsg_device_<ContainerAllocator> >
{
  static const char* value()
  {
    return "fb7e43db7d28a3165b8781af2a5c7ea9";
  }

  static const char* value(const ::xpkg_comm::xmsg_device_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xfb7e43db7d28a316ULL;
  static const uint64_t static_value2 = 0x5b8781af2a5c7ea9ULL;
};

template<class ContainerAllocator>
struct DataType< ::xpkg_comm::xmsg_device_<ContainerAllocator> >
{
  static const char* value()
  {
    return "xpkg_comm/xmsg_device";
  }

  static const char* value(const ::xpkg_comm::xmsg_device_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::xpkg_comm::xmsg_device_<ContainerAllocator> >
{
  static const char* value()
  {
    return "uint8 dev_class\n"
"uint8 dev_type\n"
"uint8 dev_number\n"
"uint8 dev_enable\n"
;
  }

  static const char* value(const ::xpkg_comm::xmsg_device_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::xpkg_comm::xmsg_device_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.dev_class);
      stream.next(m.dev_type);
      stream.next(m.dev_number);
      stream.next(m.dev_enable);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct xmsg_device_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::xpkg_comm::xmsg_device_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::xpkg_comm::xmsg_device_<ContainerAllocator>& v)
  {
    s << indent << "dev_class: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.dev_class);
    s << indent << "dev_type: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.dev_type);
    s << indent << "dev_number: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.dev_number);
    s << indent << "dev_enable: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.dev_enable);
  }
};

} // namespace message_operations
} // namespace ros

#endif // XPKG_COMM_MESSAGE_XMSG_DEVICE_H