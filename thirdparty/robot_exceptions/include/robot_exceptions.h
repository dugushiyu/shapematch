#ifndef ROBOTEXCEPTIONS_H
#define ROBOTEXCEPTIONS_H

#include <string>
#include <unordered_map>
#include <vector>


// 机器人异常管理类
class RobotExceptions {
 public:
  // 模块定义
  enum Module { HARDWARE, SOFTWARE };

  // 子模块的定义
  enum Submodule {
    OUTWARD_DETECTION_NODE
  };

  // 异常类型的定义
  enum ExceptionType {
    SURF_MATCH_NODE
  };

  // 异常行为的定义
  enum ExceptionAction { RAISE, SOLVE };

  // 异常等级的定义，如果选择默认将自动填充
  enum ExceptionLevel { DEFAULT_LEVEL, INFO_LEVEL, WARNING_LEVEL, ERROR_LEVEL };

  // 异常信息节点
  struct ExceptionNode {
    Module module;
    Submodule submodule;
    ExceptionType exception_type;
    ExceptionAction exception_action = RAISE;
    ExceptionLevel exception_level = DEFAULT_LEVEL;
    // 重载<操作符号，此类能作为std::map,std::set的键值
    bool operator<(const ExceptionNode& another_exception) const {
      if (this->exception_type == another_exception.exception_type) {
        return this->exception_level < another_exception.exception_level;
      } else {
        return this->exception_type < another_exception.exception_type;
      }
    }
    // 重载==符号,以便做比较
    bool operator==(const ExceptionNode& another_exception) const {
      if (this->module != another_exception.module) {
        return false;
      }
      if (this->submodule != another_exception.submodule) {
        return false;
      }
      if (this->exception_type != another_exception.exception_type) {
        return false;
      }
      if (this->exception_action != another_exception.exception_action) {
        return false;
      }
      if (this->exception_level != another_exception.exception_level) {
        return false;
      }
      return true;
    }
  };

  RobotExceptions();
  ~RobotExceptions();

  // 枚举转化为对应的字符串
  std::string ModuleToString(const Module& module);
  std::string SubmoduleToString(const Submodule& submodule);
  std::string ExceptionTypeToString(const ExceptionType& exception_type);
  std::string ExceptionActionToString(const ExceptionAction& exception_action);
  std::string ExceptionLevelToString(const ExceptionLevel& exception_level);

  // 字符串转化为对应的枚举值
  Module StringToModule(const std::string& module_string);
  Submodule StringToSubmodule(const std::string& submodule_string);
  ExceptionType StringToExceptionType(const std::string& exception_type_string);
  ExceptionAction StringToExceptionAction(
      const std::string& exception_action_string);
  ExceptionLevel StringToExceptionLevel(
      const std::string& exception_level_string);
  // 获取异常类型对应的默认异常等级
  ExceptionLevel DefaultExceptionLevel(const ExceptionType& exception_type);

  // 字符串英文转中文
  std::string EnglishToChinese(const std::string& english_name);
  // 字符串中文转英文
  std::string ChineseToEnglish(const std::string& chinese_name);

  // 编码生成一个异常的字符串
  std::string EncodeException(const ExceptionNode& exception_node);
  // 解码获取异常信息
  bool DecodeException(const std::string& robot_exception_string,
                       ExceptionNode* exception_node);

  // 读取自定义的异常类型对应异常等级的json文件，支持自定义异常类型的等级
  bool LoadExceptionTypeToExceptionLevel(const std::string& file_path);

 private:
  // 获取相应字符串在字符串列表中的序列号
  int GetStringIndex(const std::vector<std::string>& string_list,
                     const std::string& target_string);

 private:
  // 模块对应的字符串
  static std::vector<std::string> module_string_list_;
  // 子模块对应的字符串
  static std::vector<std::string> submodule_string_list_;
  // 异常类型对应的字符串
  static std::vector<std::string> exception_type_string_list_;
  // 异常行为对应的字符串
  static std::vector<std::string> exception_action_string_list_;
  // 异常等级对应的字符串
  static std::vector<std::string> exception_level_string_list_;
  // 异常类型对应的异常等级映射表
  static std::unordered_map<std::string, std::string>
      exception_type_to_exception_level_map_;
};

#endif  // ROBOTEXCEPTIONS_H
