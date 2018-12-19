package weka.classifiers.rules.sortinghandler;

/**
 *  This code is from the book:
*
*    Winder, R and Roberts, G (1998) <em>Developing Java
*    Software</em>, John Wiley & Sons.
*
*  It is copyright (c) 1997 Russel Winder and Graham Roberts.
*/

import weka.classifiers.rules.sortinghandler.Record ;
import weka.classifiers.rules.sortinghandler.RecordWriter ;

import java.io.BufferedWriter ;
import java.io.File ;
import java.io.FileWriter ;
import java.io.IOException ;

public class MyRecordWriter
   extends BufferedWriter
   implements RecordWriter
{
   public MyRecordWriter(File file) throws IOException
   {
       super(new FileWriter(file)) ;
   }
   
   public MyRecordWriter(String fileName) throws IOException
   {
       super(new FileWriter(fileName)) ;
   }
   
   public void writeRecord(Record r) throws IOException
   {
       write(r.toString(), 0, r.toString().length()) ;
       newLine() ;
   }

   public void finalize() throws IOException
   {
       close() ;
   }
}

