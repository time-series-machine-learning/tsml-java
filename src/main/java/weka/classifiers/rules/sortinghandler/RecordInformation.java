package weka.classifiers.rules.sortinghandler;

/**
 *  This code is from the book:
 *
 *    Winder, R and Roberts, G (1998) <em>Developing Java
 *    Software</em>, John Wiley & Sons.
 *
 *  It is copyright (c) 1997 Russel Winder and Graham Roberts.
 */


import java.io.File ;
import java.io.IOException ;

/**
 *  An interface defining a factory object encapsulating
 *  <code>Record</code> management information.  Necessary for using
 *  any <code>FileSort</code> classes.
 *
 *  <p>When dealing with files, there have to be records. This
 *  interface defines that which needs to be known in order to use any
 *  <code>FileSort</code> function objects.  The user must supply a
 *  <code>RecordInformation</code> conformant object in order to
 *  provide all the tools needed for those <code>FileSort</code>s to
 *  work.
 *
 *  @see FileSort
 *  @see Record
 *  @see RecordReader
 *  @see RecordWriter
 *  @version 1.0 19.5.97
 *  @author Russel Winder
 */
public interface RecordInformation
{
    /**
     *  We must be able to get a <code>Comparator</code> so that we
     *  can test the order of records.  Usually this will be an
     *  ordering defined by some key in the record.
     */
    Comparator getComparator() ;

    /**
     *  We must be able to get a <code>BufferedReader</code> so that
     *  we can read records from a file.
     */
    RecordReader newRecordReader(File f) throws IOException ;

    /**
     *  We must be able to get a <code>BufferedWriter</code> so that
     *  we can write records to a file.
     */
    RecordWriter newRecordWriter(File f) throws IOException ;
}

