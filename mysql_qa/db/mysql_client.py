"""
MySQL 数据库客户端

负责连接管理、表创建等数据库操作。

Usage:
    from mysql_qa.db.mysql_client import MysqlClient
    client = MysqlClient()
    client.create_table()
"""

import pymysql
from base.config import config
from base.logger import setup_logger

# 模块级 logger
logger = setup_logger('fin_rag.mysql_client')


class MysqlClient:
    """MySQL 数据库客户端，封装连接与基础操作。"""

    def __init__(self):
        """初始化数据库连接，参数从 config.ini [mysql] section 加载。"""
        self.conn = pymysql.connect(
            host=config.mysql_host,
            port=config.mysql_port,
            user=config.mysql_user,
            password=config.mysql_password,
            database=config.mysql_database,
            charset=config.mysql_charset,
            cursorclass=pymysql.cursors.DictCursor,
        )

    def create_table(self, table_name: str = 'qa_records'):
        """
        创建问答记录表。

        字段：
            id        INT         主键，自增
            question  VARCHAR(255)   问题
            answer    VARCHAR(5070)  回答

        Args:
            table_name: 表名，默认 qa_records
        """
        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id          INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
            question    VARCHAR(255)  NOT NULL DEFAULT '' COMMENT '问题',
            answer      VARCHAR(5070) NOT NULL DEFAULT '' COMMENT '回答',
            update_time DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '修改时间'
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='问答记录表';
        """
        with self.conn.cursor() as cursor:
            cursor.execute(sql)
        self.conn.commit()

    def ensure_question_unique_index(self, table_name: str = 'qa_records'):
        """确保 question 有唯一索引，支持 FAQ 数据幂等导入。"""
        with self.conn.cursor() as cursor:
            cursor.execute(
                f"SHOW INDEX FROM {table_name} WHERE Key_name = 'uq_question'"
            )
            if cursor.fetchone() is None:
                cursor.execute(
                    f"ALTER TABLE {table_name} "
                    "ADD UNIQUE KEY uq_question (question)"
                )
        self.conn.commit()

    def get_answer_by_question(self, question: str, table_name: str = 'qa_records') -> dict | None:
        """
        根据问题查询答案（精确匹配）。

        Args:
            question:   问题文本
            table_name: 表名，默认 qa_records

        Returns:
            匹配成功时返回 {'id': ..., 'question': ..., 'answer': ...}
            未匹配或异常时返回 None
        """
        sql = f"SELECT id, question, answer, update_time FROM {table_name} WHERE question = %s"
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql, (question,))
                row = cursor.fetchone()
            if row:
                logger.info("命中缓存 - id:%s question:%s", row['id'], question[:50])
            else:
                logger.debug("未命中缓存 - question:%s", question[:50])
            return row
        except pymysql.MySQLError as e:
            logger.error("查询答案失败 - question:%s error:%s", question[:50], e)
            return None

    def get_all_questions(self, table_name: str = 'qa_records') -> list[dict]:
        """
        查询表中所有问答记录。

        Args:
            table_name: 表名，默认 qa_records

        Returns:
            记录列表，每条为 {'id': ..., 'question': ..., 'answer': ...}
            异常时返回空列表 []
        """
        sql = f"SELECT id, question, answer, update_time FROM {table_name}"
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
            logger.info("查询所有问题 - 共 %d 条记录", len(rows))
            return rows
        except pymysql.MySQLError as e:
            logger.error("查询所有问题失败 - error:%s", e)
            return []

    def batch_insert(self, records: list[tuple[str, str]], table_name: str = 'qa_records') -> int:
        """
        批量插入问答记录。

        Args:
            records:    记录列表，每条为 (question, answer) 元组
            table_name: 表名，默认 qa_records

        Returns:
            成功插入的行数，异常时返回 0
        """
        if not records:
            logger.warning("批量插入 - 数据为空，跳过")
            return 0

        sql = f"INSERT INTO {table_name} (question, answer) VALUES (%s, %s)"
        try:
            with self.conn.cursor() as cursor:
                affected = cursor.executemany(sql, records)
            self.conn.commit()
            logger.info("批量插入成功 - 共 %d 条记录", affected)
            return affected
        except pymysql.MySQLError as e:
            self.conn.rollback()
            logger.error("批量插入失败 - error:%s", e)
            return 0

    def batch_upsert(self, records: list[tuple[str, str]], table_name: str = 'qa_records') -> int:
        """批量写入 FAQ；问题已存在时更新答案。"""
        if not records:
            return 0

        sql = (
            f"INSERT INTO {table_name} (question, answer) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE answer = VALUES(answer)"
        )
        try:
            with self.conn.cursor() as cursor:
                cursor.executemany(sql, records)
            self.conn.commit()
            logger.info("FAQ 批量写入完成 - 共 %d 条", len(records))
            return len(records)
        except pymysql.MySQLError as e:
            self.conn.rollback()
            logger.error("FAQ 批量写入失败 - error:%s", e)
            raise

    def close(self):
        """关闭数据库连接。"""
        if self.conn:
            self.conn.close()


_client_instance: MysqlClient | None = None


def get_mysql_client() -> MysqlClient:
    """懒加载 MySQL 客户端，避免模块导入阶段因数据库暂不可用而崩溃。"""
    global _client_instance
    if _client_instance is None:
        _client_instance = MysqlClient()
    return _client_instance


if __name__ == '__main__':
    get_mysql_client().create_table()
