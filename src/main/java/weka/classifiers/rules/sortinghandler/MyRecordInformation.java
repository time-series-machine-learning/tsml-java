package weka.classifiers.rules.sortinghandler;

/**
 *  This code is from the book:
 *
 *    Winder, R and Roberts, G (1998) <em>Developing Java
 *    Software</em>, John Wiley & Sons.
 *
 *  It is copyright (c) 1997 Russel Winder and Graham Roberts.
 */

import weka.classifiers.rules.sortinghandler.Comparator ;
import weka.classifiers.rules.sortinghandler.RecordInformation ;
import weka.classifiers.rules.sortinghandler.RecordReader ;
import weka.classifiers.rules.sortinghandler.RecordWriter ;

import java.io.File ;
import java.io.IOException ;

public class MyRecordInformation implements RecordInformation
{
    public MyRecordInformation(Comparator r)
    {
        c = r ;
    }

    public Comparator getComparator()
    {
        return c ;
    }
    
    public RecordReader newRecordReader(File f) throws IOException
    {
        return new MyRecordReader(f) ;
    }
    
    public RecordWriter newRecordWriter(File f) throws IOException
    {
        return new MyRecordWriter(f) ;
    }

    private Comparator c ;
}

