-- MySQL dump 10.13  Distrib 5.5.31, for debian-linux-gnu (x86_64)
--
-- Host: 10.64.16.32    Database: enwiki
-- ------------------------------------------------------
-- Server version	5.5.34-MariaDB-1$1precise-log

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `externallinks`
--

DROP TABLE IF EXISTS `externallinks`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `externallinks` (
  `el_from` int(8) unsigned NOT NULL DEFAULT '0',
  `el_to` blob NOT NULL,
  `el_index` blob NOT NULL,
  KEY `el_from` (`el_from`,`el_to`(40)),
  KEY `el_to` (`el_to`(60),`el_from`),
  KEY `el_index` (`el_index`(60))
) ENGINE=MyISAM DEFAULT CHARSET=binary;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `externallinks`
--

/*!40000 ALTER TABLE `externallinks` DISABLE KEYS */;

LOAD DATA LOCAL INFILE '$1/enwiki-$2-externallinks.main.tsv' IGNORE INTO TABLE externallinks CHARACTER SET binary FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\''; 
/*!40000 ALTER TABLE `externallinks` ENABLE KEYS */;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = '' */ ;
DELIMITER ;;
/*!50003 CREATE*/ /*!50017 DEFINER=`root`@`10.%`*/ /*!50003 trigger externallinks_insert after insert on externallinks
  for each row begin
    
    insert into externallinks_new (
      el_hash,
      el_from,
      el_to,
      el_index
    )
    values (
      md5(concat(
        cast(ifnull(NEW.el_from,'(null)') as char),':',
        cast(ifnull(NEW.el_to,'(null)') as char),':',
        cast(ifnull(NEW.el_index,'(null)') as char)
      )),
      NEW.el_from,
      NEW.el_to,
      NEW.el_index
    );

    update externallinks_counters set counter = counter + 1 where name = 'inserts';

  end */;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = '' */ ;
DELIMITER ;;
/*!50003 CREATE*/ /*!50017 DEFINER=`root`@`10.%`*/ /*!50003 trigger externallinks_update after update on externallinks
  for each row begin

    update externallinks_new set
    el_hash = md5(concat(
      cast(ifnull(NEW.el_from,'(null)') as char),':',
      cast(ifnull(NEW.el_to,'(null)') as char),':',
      cast(ifnull(NEW.el_index,'(null)') as char)
    )),
    el_from  = NEW.el_from,
    el_to    = NEW.el_to,
    el_index = NEW.el_index
    where el_hash = md5(concat(
      cast(ifnull(OLD.el_from,'(null)') as char),':',
      cast(ifnull(OLD.el_to,'(null)') as char),':',
      cast(ifnull(OLD.el_index,'(null)') as char)
    ));
  
    update externallinks_counters set counter = counter + 1 where name = 'updates';

  end */;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!50003 SET @saved_cs_client      = @@character_set_client */ ;
/*!50003 SET @saved_cs_results     = @@character_set_results */ ;
/*!50003 SET @saved_col_connection = @@collation_connection */ ;
/*!50003 SET character_set_client  = utf8 */ ;
/*!50003 SET character_set_results = utf8 */ ;
/*!50003 SET collation_connection  = utf8_general_ci */ ;
/*!50003 SET @saved_sql_mode       = @@sql_mode */ ;
/*!50003 SET sql_mode              = '' */ ;
DELIMITER ;;
/*!50003 CREATE*/ /*!50017 DEFINER=`root`@`10.%`*/ /*!50003 trigger externallinks_delete after delete on externallinks
  for each row begin

    delete from externallinks_new
    where el_hash = md5(concat(
      cast(ifnull(OLD.el_from,'(null)') as char),':',
      cast(ifnull(OLD.el_to,'(null)') as char),':',
      cast(ifnull(OLD.el_index,'(null)') as char)
    ));
  
    update externallinks_counters set counter = counter + 1 where name = 'deletes';

  end */;;
DELIMITER ;
/*!50003 SET sql_mode              = @saved_sql_mode */ ;
/*!50003 SET character_set_client  = @saved_cs_client */ ;
/*!50003 SET character_set_results = @saved_cs_results */ ;
/*!50003 SET collation_connection  = @saved_col_connection */ ;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2014-01-02 23:19:46
