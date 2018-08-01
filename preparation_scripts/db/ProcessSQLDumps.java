/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.Writer;
import java.util.AbstractMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 *
 * @author sajadi
 */
public class ProcessSQLDumps {

    public static String wikiver;
    public static String directory;

    public static String pagefilename;
    public static String mainpagefilename;

    public static String linksfilename;
    public static String mainlinksfilename;
    public static String deadlinksfilename;

    public static String redirsfilename;
    public static String mainredirsfilename;
    public static String deadredirsfilename;

    public static String catfilename;
    public static String maincatfilename;
    public static String deadcatfilename;

    public static String catlinksfilename;
    public static String maincatlinksfilename;
    public static String deadcatlinksfilename;

    public static String extlinksfilename;
    public static String mainextlinksfilename;

    public static String intp;
    public static String intpn;
    public static String stringp;
    public static String stringpn;
    public static String doublep;
    public static String doublepn;
    
    public static String printElapsedTime(long miliseconds) {
        long h = TimeUnit.MILLISECONDS.toHours(miliseconds);
        long m = TimeUnit.MILLISECONDS.toMinutes(miliseconds) - TimeUnit.HOURS.toMinutes(h);
        long s = TimeUnit.MILLISECONDS.toSeconds(miliseconds) - TimeUnit.MINUTES.toSeconds(m)
                - TimeUnit.HOURS.toSeconds(h);
        return String.format("Elapsed Time = %d:%d:%d", h, m, s);
    }
    
    public static String removeEnd(String s, int i) {
        
        return s.substring(0, s.length() - i);
    }
    
    public static Scanner getReader(String filename) throws Exception{
        Scanner scanner = new Scanner(new BufferedReader(new InputStreamReader(new FileInputStream(filename), "ISO-8859-1")));
        return scanner;
    }
    
    public static Writer getWriter(String filename) throws Exception{
        Writer writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "ISO-8859-1"),32768);
        return writer;
    }
    public static class Page{
        int id;
        int namespace;
        String title;
        int isRedirect;

        public Page(int id, int namespace, String title, int isRedirect) {
            this.id = id;
            this.namespace = namespace;
            this.title = title;
            this.isRedirect = isRedirect;
        }
    }
    
    // Not that pageIds includes catIds aswell
    public static Set<Integer> pageIds; 
    public static Set<Integer> catIds;
    
    public static Set <Integer> PagesMarkedRedir;
    public static List<Page> pageStrings;
    public static Map<String, Integer> title2id;
    public static Map<String, Integer> cat2id;

    
    public static Map<Integer,Integer> redirects;
    public static List<Integer> redirIds;
    public static Set<Integer> invalids;

    public static void init() {
        intp = "(?:(?:-|\\+)?[0-9]+)";
        intpn = String.format("(?:%s|(?:NULL))?",intp);
        stringp = "(?:'(?:[^\\\\']|(?:\\\\')|(?:\\\\\")|(?:\\\\\\\\))*')";
        stringpn = String.format("(?:%s|(?:NULL))",stringp);
        doublep = String.format("(?:(?:%s?(?:\\.)?[0-9]+)(?:e%s)?)",intp,intp);
        doublepn = String.format("(?:%s|(?:NULL))?",doublep);
    }
    public static void init(String[] args) throws Exception {
       int i=-1;
        pagefilename=null;
        linksfilename=null;
        redirsfilename=null;
        wikiver=null;
        directory = args[0];
        if (!directory.endsWith("/"))
            directory +='/';
        File dir = new File(directory);
        Pattern pattern = Pattern.compile("enwiki-(.*)-(.*)\\.sql$");
        for (File f: dir.listFiles()){
            String fname= f.getAbsolutePath();
            Matcher matcher = pattern.matcher(fname);
            if (matcher.find()){
                if (wikiver==null)
                    wikiver = matcher.group(1);
                else if (!matcher.group(1).equals(wikiver)){
                     throw new Exception("dump files should have the same version");
                }
                String tablename = matcher.group(2);
            
                if (tablename.equals("page")){
                    pagefilename = fname; 
                    mainpagefilename = directory+"enwiki-"+wikiver+"-"+tablename +".main.tsv";
                }                        

                if (tablename.equals("pagelinks")){
                    linksfilename = fname;
                    mainlinksfilename = directory+"enwiki-"+wikiver + "-pagelinks.main.tsv";
                    deadlinksfilename = directory+"enwiki-"+wikiver + "-pagelinks.dead.tsv";
                }       

                if (tablename.equals("redirect")){
                    redirsfilename = fname;
                    mainredirsfilename = directory+"enwiki-"+wikiver + "-redirect.main.tsv";
                    deadredirsfilename = directory+"enwiki-"+wikiver + "-redirect.dead.tsv";
                }

                if (tablename.equals("category")){
                    catfilename = fname;
                    maincatfilename = directory+"enwiki-"+wikiver+ "-category.main.tsv";
                    deadcatfilename = directory+"enwiki-"+wikiver+ "-category.dead.tsv";
                }

                if (tablename.equals("categorylinks")){
                    catlinksfilename = fname;
                    maincatlinksfilename = directory+"enwiki-"+wikiver+"-categorylinks.main.tsv";
                    deadcatlinksfilename = directory+"enwiki-"+wikiver+"-categorylinks.dead.tsv";
                }
                if (tablename.equals("externallinks")){
                    extlinksfilename = fname;
                    mainextlinksfilename = directory+"enwiki-"+wikiver+"-"+tablename + ".main.tsv";
                }
            }
        }
//        if (pagefilename==null){
//            throw new FileNotFoundException("pagefile not found");
//        }    
//        if (redirsfilename==null){
//            throw new FileNotFoundException("redirect file not found");
//        }    
//        if (linksfilename==null){
//            throw new FileNotFoundException("link file not found");
//        }    
        init();
    }

    public static void processpages() throws Exception {
        System.out.print("Processing pages...\n");
        Scanner page = getReader(pagefilename);


        pageIds = new HashSet<Integer>();
        catIds = new HashSet<Integer>();
        PagesMarkedRedir=new HashSet<Integer> ();
        title2id = new HashMap<String, Integer>();
        cat2id = new HashMap<String, Integer>();
        pageStrings=new LinkedList<Page>();
        
        

        String pageregex =  String.format("\\((%s),(%s),(%s),%s,%s,(%s),%s,%s,%s,%s,%s,%s,%s\\)(,|;)", intp, intp, stringp, stringp, intp, intp, intp, doublep, stringp, stringpn, intp, intp, stringpn);
        if (wikiver.equals("20140102")) {
            pageregex =     String.format("\\((%s),(%s),(%s),%s,%s,(%s),%s,%s,%s,%s,%s,%s\\)(,|;)", intp, intp, stringp, stringp, intp, intp, intp, doublep, stringp, stringpn, intp, intp);
        }
        if (wikiver.equals("20120403")) {
            pageregex =     String.format("\\((%s),(%s),(%s),%s,%s,(%s),%s,%s,%s,%s,%s\\)(,|;)", intp, intp, stringp, stringp, intp, intp, intp, doublep, stringp, intp, intp);
        }
        Pattern pattern = Pattern.compile(pageregex);
        page.useDelimiter("\\n");
        while (page.hasNext()) {
            String line = page.next();
            if (!line.startsWith("INSERT INTO")) {
                continue;
            }
            line = line.trim();
            Matcher matcher = pattern.matcher(line);
            while (matcher.find()) {
                int id=Integer.parseInt(matcher.group(1));
                int namespace=Integer.parseInt(matcher.group(2));
                String title=matcher.group(3);
                int isRedirect=Integer.parseInt(matcher.group(4));
                if (namespace==0 || namespace==14){
                    pageIds.add(id);
                    if (namespace==14){
                        catIds.add(id);                        
                    }
                    if (isRedirect==1)
                        PagesMarkedRedir.add(id);
                    pageStrings.add(new Page(id, namespace, title, isRedirect));

                    if (namespace==0) {
                        title2id.put(title, id);
                    } else if (namespace==14) {
                        cat2id.put(title, id);
                    }
                 }
            }

        }
        page.close();
        System.out.println("Done...\n");

    }

    public static void processredirs() throws Exception {
        System.out.print("Processing redirects...\n");
        
        Scanner redirs = getReader(redirsfilename);

        Writer deadredirs = getWriter(deadredirsfilename);
        
        redirects=new HashMap<Integer, Integer>();
        redirIds=new LinkedList<Integer>();
        String redirsregex = String.format("\\((%s),(%s),(%s),%s,%s\\)(,|;)", intp, intp, stringp,stringp,stringp);
        Pattern pattern = Pattern.compile(redirsregex);
        redirs.useDelimiter("\\n");
        while (redirs.hasNext()) {
            String line = redirs.next();
            if (!line.startsWith("INSERT INTO")) {
                continue;
            }
            line = line.trim();
            Matcher matcher = pattern.matcher(line);
            while (matcher.find()) {
                int rdFrom=Integer.parseInt(matcher.group(1));
                int namespace=Integer.parseInt(matcher.group(2));
                String rdTo=matcher.group(3);

                if ((namespace==0 || namespace==14) && pageIds.contains(rdFrom)) {
                    if (namespace==0){
                        if (title2id.containsKey(rdTo)) {
                            redirects.put(rdFrom, title2id.get(rdTo));
                            redirIds.add(rdFrom);
                        }else {
                            deadredirs.write(String.format("%d,%d,%s:invalid title\n", rdFrom, namespace, rdTo));
                        }
                    } else if (namespace==14){
                        if (cat2id.containsKey(rdTo)) {
                            redirects.put(rdFrom, cat2id.get(rdTo));
                            redirIds.add(rdFrom);
                        }else {
                            deadredirs.write(String.format("%d,%d,%s:invalid title\n", rdFrom, namespace, rdTo));
                        }
                    }
                }

            }
        }
        deadredirs.close();
        redirs.close();
        System.out.println("Done...\n");

    }
    
    
    public static Integer getTargetPage(Integer p){
        if (invalids.contains(p))
            return null;

        Set<Integer> visiteds = new HashSet<Integer>();
        
        visiteds.add(p);
        while (redirects.containsKey(p)){
            if (invalids.contains(p))
                return null;
            p = redirects.get(p);
            if (p==null)
                return null;
            if (visiteds.contains(p)) {
                return null;
            }
            else {
                visiteds.add(p);
            }
        }
        return p;
    }    
    
    public static void resolveRedirs() throws Exception{
        System.out.println("Resolving redirs");
        invalids=new HashSet<Integer>();
        
        long start, elapsedTimeMillis;
        
        PrintWriter pw= new PrintWriter(directory+"enwiki-"+wikiver+"-redirrep.tsv");


        //iteration 1
        pw.println("Iteration 1 started");
        start = System.currentTimeMillis();
        
        for (int id:PagesMarkedRedir){
            if(!redirects.containsKey(id)){
                invalids.add(id);
                pw.println(id);
            }
        }
        elapsedTimeMillis = System.currentTimeMillis() - start;
        pw.println("done in:" + printElapsedTime(elapsedTimeMillis));
        
        //iteration 2        
        pw.println("Iteration 2 started");
        start = System.currentTimeMillis();

        for (int id:redirects.keySet()){            
            if(!PagesMarkedRedir.contains(id)){
                invalids.add(id);
                pw.println(id);
            }
        }

        
        elapsedTimeMillis = System.currentTimeMillis() - start;
        pw.println("done in:" + printElapsedTime(elapsedTimeMillis));

        //iteration 3
        pw.println("Iteration 3 started");       
        start = System.currentTimeMillis();

        for (Map.Entry<Integer,Integer> entry:redirects.entrySet()){
            Integer newValue=getTargetPage(entry.getValue());
            if (newValue==null){
                invalids.add(entry.getKey());
                pw.println(entry.getKey() + "\t"+entry.getValue()+" -> null");
                continue;
            }
            if (newValue.equals(entry.getValue()))
                continue;
            pw.println(entry.getKey() + "\t"+entry.getValue()+" -> "+ newValue);
            entry.setValue(newValue);
        }
        elapsedTimeMillis = System.currentTimeMillis() - start;
        pw.println("done in:" + printElapsedTime(elapsedTimeMillis));
        
        
        pw.close();
        
        
        
    }
    public static void removeinvalid(){
//      
//    
        System.out.println("Removing invalids");
    
        pageIds.removeAll(invalids);
        catIds.removeAll(invalids);
        PagesMarkedRedir.removeAll(invalids);

        
        for(Iterator<Page> it=pageStrings.iterator();it.hasNext();){
            int id=it.next().id;
            if (invalids.contains(id)){
                it.remove();
            }
        }
        
        title2id.values().removeAll(invalids);
        cat2id.values().removeAll(invalids);
        
        redirects.keySet().removeAll(invalids);
        redirIds.removeAll(invalids);
        
    }
    public static void writePagesRedirs() throws Exception{
        System.out.println("Writing pages...");
        Writer mainpage = getWriter(mainpagefilename);
        
        Iterator<Page> it=pageStrings.iterator();
        
        for (Page p:pageStrings){
            mainpage.write(String.format("%d,%d,%s,%d\n", p.id, p.namespace, p.title, p.isRedirect));  
        }
        mainpage.close();
        
        System.out.println("Writing redirects...");
        Writer mainredirs = getWriter(mainredirsfilename);
        for (int i:redirIds){
            mainredirs.write(String.format("%d,%d\n", i, redirects.get(i)));  
        }       
        mainredirs.close();
        
    }
    public static Integer getSyn(Integer id){
        Integer syn=redirects.get(id);
        if (redirects.containsKey(id))
            return redirects.get(id);
        return id;
    }
    public static boolean isDuplicate(Integer src, Integer dest, Set<AbstractMap.SimpleEntry<Integer,Integer>> linksSet){
//        AbstractMap.SimpleEntry<Integer,Integer> link=new AbstractMap.SimpleEntry<Integer, Integer>(src, dest);
//        if (linksSet.contains(link))
//            return true;
//        linksSet.add(link);

        return src.equals(dest);
    }
    public static void processpagelinks() throws Exception {
        System.out.print("Processing pagelinks...\n");
        
        Set<AbstractMap.SimpleEntry<Integer,Integer>> linksSet=new HashSet<AbstractMap.SimpleEntry<Integer,Integer>>();
        Scanner links=getReader(linksfilename);
        Writer mainlinks = getWriter(mainlinksfilename);
        Writer deadlinks = getWriter(deadlinksfilename);
        boolean isnewver = true;    
        String linksregex =     String.format("\\((%s),(%s),(%s),(%s)\\)(,|;)", intp, intp, stringp, intp);
        if (wikiver.equals("20140102")) {
            isnewver = false;
            linksregex =        String.format("\\((%s),(%s),(%s)\\)(,|;)", intp, intp, stringp);
        }
        Pattern pattern = Pattern.compile(linksregex);
        links.useDelimiter("\\n");
        while (links.hasNext()) {
            String line = links.next();
            if (!line.startsWith("INSERT INTO")) {
                continue;
            }
            line = line.trim();
            Matcher matcher = pattern.matcher(line);
            while (matcher.find()) {
                int plFrom=Integer.parseInt(matcher.group(1));
                int namespace=Integer.parseInt(matcher.group(2));
                String plTo=matcher.group(3);
                int plFromNamespace = isnewver ? Integer.parseInt(matcher.group(4)): -1;
                if (!pageIds.contains(plFrom)){
                    if (isnewver && (plFromNamespace ==0 || plFromNamespace ==14))
                        deadlinks.write(String.format("namespace mismatch: %d,%d,%s,%d\n", plFrom, namespace, plTo, plFromNamespace));
                    continue;
                }
                // So for sure a page or cat
                if (isnewver){
                    if ((catIds.contains(plFrom) && plFromNamespace != 14) || 
                       (!catIds.contains(plFrom) && plFromNamespace != 0)){
                    	deadlinks.write(String.format("namespace mismatch: %d,%d,%s,%d\n", plFrom, namespace, plTo, plFromNamespace));
                    	continue;
		    }
                }
                if (namespace==0) {
                    if (title2id.containsKey(plTo)) {
                        int plSrc=getSyn(plFrom);
                        int plDst=getSyn(title2id.get(plTo));
                        if (isDuplicate(plSrc, plDst, linksSet )){
                            deadlinks.write(String.format("Dup: %d,%d,%s\n", plFrom, namespace, plTo));                            
                            continue;
                        }
                        mainlinks.write(String.format("%d,%d\n", plSrc, plDst));
                    } else {
                        deadlinks.write(String.format("%d,%d,%s,%d\n", plFrom, namespace, plTo, plFromNamespace));
                    }
                } else if (namespace==14) {
                    if (cat2id.containsKey(plTo)) {
                        int plSrc=getSyn(plFrom);
                        int plDst=getSyn(cat2id.get(plTo));
                        if (isDuplicate(plSrc, plDst, linksSet)){
                            deadlinks.write(String.format("Dup: %d,%d,%s,%d\n", plFrom, namespace, plTo, plFromNamespace));                            
                            continue;
                        }
                        mainlinks.write(String.format("%d,%d\n", plSrc, plDst));
                    } else {
                        deadlinks.write(String.format("%d,%d,%s,%d\n", plFrom, namespace, plTo, plFromNamespace));
                    }
                }
            }

        }
        deadlinks.close();
        mainlinks.close();
        links.close();
        System.out.println("Done...\n");

    }
    
    public static void processcategories() throws Exception {
        System.out.print("Processing categories...\n");

        Scanner cat= getReader(catfilename);
        Writer maincat = getWriter(maincatfilename);

        String catregex = String.format("\\((%s),(%s),(%s),(%s),%s\\)(,|;)", intp, stringp,intp,intp,intp);
        Pattern pattern = Pattern.compile(catregex);
        cat.useDelimiter("\\n");
        while (cat.hasNext()) {
            String line = cat.next();
            if (!line.startsWith("INSERT INTO")) {
                continue;
            }
            line = line.trim();
            Matcher matcher = pattern.matcher(line);
            while (matcher.find()) {
                maincat.write(String.format("%s,%s,%s,%s\n", matcher.group(1), matcher.group(2),matcher.group(3),matcher.group(4)));
            }

        }
        maincat.close();
        cat.close();
        System.out.println("Done...\n");

    }    
    public static void processcatlinks() throws Exception {
        System.out.print("Processing catlinks...\n");

        Set<AbstractMap.SimpleEntry<Integer,Integer>> linksSet=new HashSet<AbstractMap.SimpleEntry<Integer,Integer>>();
        Scanner catlinks= getReader(catlinksfilename);
        
        Writer maincatlinks = getWriter(maincatlinksfilename);
        Writer deadcatlinks = getWriter(deadcatlinksfilename);

        String catlinksregex = String.format("\\((%s),(%s),%s,%s,%s,%s,(%s)\\)(,|;)", intp,stringp, stringp, stringp, stringp, stringp, stringp);
        Pattern pattern = Pattern.compile(catlinksregex);
        catlinks.useDelimiter("\\n");
        while (catlinks.hasNext()) {
            String line = catlinks.next();
            if (!line.startsWith("INSERT INTO")) {
                continue;
            }
            line = line.trim();
            Matcher matcher = pattern.matcher(line);
            while (matcher.find()) {
                int clFrom=Integer.parseInt(matcher.group(1));
                String clTitle=matcher.group(2);
                String clType=matcher.group(3);

                if (pageIds.contains(clFrom)) {
                    if (cat2id.containsKey(clTitle)) {
                        int clSrc=getSyn(clFrom);
                        int clDst=getSyn(cat2id.get(clTitle));
                        if (isDuplicate(clSrc, clDst, linksSet)){
                            deadcatlinks.write(String.format("Dup: %d,%s,%s\n", clFrom, clTitle, clType));                            
                            continue;
                        }
                        maincatlinks.write(String.format("%d,%d,%s\n", clSrc, clDst,clType));
                    } else {
                        deadcatlinks.write(String.format("%d,%s,%s\n", clFrom, clTitle, clType));
                    }
                }
            }

        }
        deadcatlinks.close();
        maincatlinks.close();
        catlinks.close();
        System.out.println("Done...\n");

    }
    public static void processexterallinks() throws Exception {
        System.out.print("Processing extlinks...\n");

        Scanner extlinks=getReader(extlinksfilename);
        Writer mainextlinks = getWriter(mainextlinksfilename);
        

        String extlinksregex = String.format("\\((%s),(%s),(%s)\\)(,|;)", intp, stringp,stringp);
        Pattern pattern = Pattern.compile(extlinksregex);
        extlinks.useDelimiter("\\n");
        while (extlinks.hasNext()) {
            String line = extlinks.next();
            if (!line.startsWith("INSERT INTO")) {
                continue;
            }
            line = line.trim();
            Matcher matcher = pattern.matcher(line);
            while (matcher.find()) {
                mainextlinks.write(String.format("%s,%s,%s\n", matcher.group(1), matcher.group(2),matcher.group(3)));
            }
        }
        mainextlinks.close();
        extlinks.close();
        System.out.println("Done...\n");

    }
    // Uncomment the commented lines for testing
    public static void unicodetest1() throws Exception{
        //Wikipedia w=Wikipedia.create();
        String[] inargs={"20140201","test/pagetestsmall.sql"};
        
        init(inargs);
        String testid=null;
        Scanner page = new Scanner(new BufferedReader(new InputStreamReader(new FileInputStream(inargs[1]),"UTF-8")));

        String pageregex = String.format("\\((%s),(%s),(%s),%s,%s,(%s),%s,%s,%s,%s,%s,%s\\)(,|;)", intp, intp, stringp, stringp, intp, intp, intp, doublep, stringp, stringpn, intp, intp);

        Pattern pattern = Pattern.compile(pageregex);
        page.useDelimiter("\\n");
        String line = new String (page.next());
        System.out.println(line);
        Matcher matcher = pattern.matcher(line);
        if (matcher.find()) {
          testid=matcher.group(3);
          testid=testid.substring(1,testid.length()-1);
        }
//        wikiapi.Page a=w.getPageByTitle(testid);
//        if (a==null)
//            System.out.println(testid+": not found");
//        else
//            System.out.println(a.getId());
        
        
        page.close();
        System.out.println("Done...\n");
        
        
        
    }
    public static void unicodetest2() throws Exception{
       
        init();
        
        
        Scanner cat = new Scanner(new BufferedReader(new InputStreamReader(new FileInputStream("test/iso8859test.txt"),"ISO-8859-1")));
        //Scanner cat = new Scanner(new BufferedReader(new InputStreamReader(new FileInputStream("test/utftest.txt"),"UTF-8")));
        BufferedInputStream bis=new BufferedInputStream(new FileInputStream("test/iso8859test.txt"));
        //BufferedInputStream bis=new BufferedInputStream(new FileInputStream("test/utftest.txt"));

        //Writer catout = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("test/utftest.out.txt"),"UTF-8"));
        Writer catout = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("test/iso8859test.out.txt"),"ISO-8859-1"));
        
        
        String catregex = String.format("%s,(%s),%s,%s", intp,stringp,intp,intp);

        Pattern pattern = Pattern.compile(catregex);
        
        cat.useDelimiter("\\n");
        String line = new String (cat.next());
        byte[] byteline1=line.getBytes("ISO-8859-1");
        
        byte[] byteline2=new byte[1000];
        int len=bis.read(byteline2);       
       
        for (int i=0;i<line.length();i++){
            if (byteline1[i]!=byteline2[i]){
                System.out.printf("%c: %d and %d\n", line.charAt(i), byteline1[i],byteline2[i]);
            }
        }        
        System.out.println(line);
        catout.write(line);
        
        bis.close();
        cat.close();
        catout.close();
        
        
        
    }
    
//    public static byte[] readline(BufferedInputStream bis){
//        ArrayList<Byte> array=new ArrayList<Byte>();
//        while(true){
//            Byte b= bis.read();
//            if (b=='\r'){
//                break;
//            }
//        }
//    }

    public static void testfun(){
        Map<Integer,Integer> m=new HashMap<Integer, Integer>();
        m.put(1, 1);
        m.put(12, null);
        for (Map.Entry<Integer,Integer> e:m.entrySet()){
            System.out.printf("%s,%s\n", e.getKey(),e.getValue());
        }
    }    
    public static void main(String[] args) throws Exception{
        //unicodetest2();
        //testfun();
         if (args.length==0){
            System.err.println("usage: java ProcessSQLDumps <dir>");
            System.err.println("\tdir:\tPath to the directory containing sql dump files");
            return;
        }
        
        long start, elapsedTimeMillis;
        
 

        //String[] inargs={"20140201","pagetest.sql","pagelinkstest.sql"};
        init(args);
        
        start = System.currentTimeMillis();
        processpages();
        elapsedTimeMillis = System.currentTimeMillis() - start;
        System.out.println(printElapsedTime(elapsedTimeMillis));
        
        
        start = System.currentTimeMillis();
        processredirs();
        elapsedTimeMillis = System.currentTimeMillis() - start;
        System.out.println(printElapsedTime(elapsedTimeMillis));
        
        start = System.currentTimeMillis();
        resolveRedirs();
        elapsedTimeMillis = System.currentTimeMillis() - start;
        System.out.println(printElapsedTime(elapsedTimeMillis));

        start = System.currentTimeMillis();
        removeinvalid();
        elapsedTimeMillis = System.currentTimeMillis() - start;
        System.out.println(printElapsedTime(elapsedTimeMillis));

        start = System.currentTimeMillis();
        writePagesRedirs();
        elapsedTimeMillis = System.currentTimeMillis() - start;
        System.out.println(printElapsedTime(elapsedTimeMillis));
        

        start = System.currentTimeMillis();
        processpagelinks();
        elapsedTimeMillis = System.currentTimeMillis() - start;
        System.out.println("done in:" + printElapsedTime(elapsedTimeMillis));
        
        if ((catfilename==null) || (catlinksfilename==null))
            return;
        start = System.currentTimeMillis();
        processcategories();
        elapsedTimeMillis = System.currentTimeMillis() - start;
        System.out.println("done in:" + printElapsedTime(elapsedTimeMillis));
    
        start = System.currentTimeMillis();
        processcatlinks();
        elapsedTimeMillis = System.currentTimeMillis() - start;
        System.out.println("done in:" + printElapsedTime(elapsedTimeMillis));

//        start = System.currentTimeMillis();
//        processexterallinks();
//        elapsedTimeMillis = System.currentTimeMillis() - start;
//        System.out.println("done in:" + printElapsedTime(elapsedTimeMillis));
    
    }
}
