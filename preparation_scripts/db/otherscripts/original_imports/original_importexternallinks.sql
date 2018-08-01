-- MySQL dump 10.13  Distrib 5.5.44, for debian-linux-gnu (x86_64)
--
-- Host: 10.64.32.23    Database: enwiki
-- ------------------------------------------------------
-- Server version	5.5.5-10.0.16-MariaDB-log

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
  `el_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  `el_from` int(8) unsigned NOT NULL DEFAULT '0',
  `el_from_namespace` int(11) NOT NULL DEFAULT '0',
  `el_to` blob NOT NULL,
  `el_index` blob NOT NULL,
  PRIMARY KEY (`el_id`),
  KEY `el_from` (`el_from`,`el_to`(40)),
  KEY `el_to` (`el_to`(60),`el_from`),
  KEY `el_index` (`el_index`(60)),
  KEY `el_backlinks_to` (`el_from_namespace`,`el_to`(60),`el_from`)
) ENGINE=InnoDB AUTO_INCREMENT=162352531 DEFAULT CHARSET=binary;
/*!40101 SET character_set_client = @saved_cs_client */;

--

Load DATA INFILE '/users/grad/sajadi/backup/projectsarchived/Datasets/wikipedia/20151002/enwiki-20151002-externallinks.main.tsv' INTO TABLE externallinks CHARACTER SET binary FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\'';

/*!40000 ALTER TABLE `externallinks` ENABLE KEYS */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2015-10-04 10:03:34
