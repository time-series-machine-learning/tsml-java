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
import weka.classifiers.rules.sortinghandler.RecordReader ;

import java.io.BufferedReader ;
import java.io.File ;
import java.io.FileReader ;
import java.io.IOException ;

public class MyRecordReader
    extends BufferedReader
    implements RecordReader
{
    public MyRecordReader(File file) throws IOException
    {
        super(new FileReader(file)) ;
    }
    
    public MyRecordReader(String fileName) throws IOException
    {
        super(new FileReader(fileName)) ;
    }
    
    public Record readRecord() throws IOException
    {
        String s = readLine() ;
        return (s == null) ? null : new MyRecord(s) ;
    }

    public void finalize() throws IOException
    {
        close() ;
    }
}

