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
-- Table structure for table `categorylinks`
--

DROP TABLE IF EXISTS `categorylinks`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `categorylinks` (
  `cl_from` int(8) unsigned NOT NULL DEFAULT '0',
  `cl_to` int(8) unsigned NOT NULL DEFAULT '0',
  `cl_type` enum('page','subcat') NOT NULL DEFAULT 'page',
  UNIQUE KEY `cl_from` (`cl_from`,`cl_to`),
  UNIQUE KEY `cl_to` (`cl_to`,`cl_from`)
) ENGINE=MyISAM DEFAULT CHARSET=binary;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `categorylinks`
--

/*!40000 ALTER TABLE `categorylinks` DISABLE KEYS */;

LOAD DATA LOCAL INFILE '$1/enwiki-$2-categorylinks.main.tsv' IGNORE INTO TABLE categorylinks CHARACTER SET binary FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\''; 
/*!40000 ALTER TABLE `categorylinks` ENABLE KEYS */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2014-01-02 21:18:40
