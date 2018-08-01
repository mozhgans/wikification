-- MySQL dump 10.13  Distrib 5.5.31, for debian-linux-gnu (x86_64)
--
-- Host: 10.64.16.32    Database: enwiki
-- ------------------------------------------------------
-- Server version	5.5.34-MariaDB-1$1precise-log


--
-- Table structure for table `pagelinks`
--

DROP TABLE IF EXISTS `pagelinksorderedin`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `pagelinksorderedin` (
  `cache_id`    int(8) unsigned NOT NULL DEFAULT '0',
  `in_neighb`   longblob NOT NULL,
  PRIMARY KEY (`cache_id`)
) ENGINE=MyISAM DEFAULT CHARSET=binary;
/*!40101 SET character_set_client = @saved_cs_client */;


