package weka.classifiers.rules.sortinghandler;

/**
 *  This code is from the book:
 *
 *    Winder, R and Roberts, G (1998) <em>Developing Java
 *    Software</em>, John Wiley & Sons.
 *
 *  It is copyright (c) 1997 Russel Winder and Graham Roberts.
 */

import java.io.FileNotFoundException ;
import java.io.IOException ;

/**
 *  The interface implemented by any file sorting function object.
 *
 *  <p><code>FileSort</code>ers assume that there is a class
 *  conforming to <code>RecordInformation</code>, defined by the user,
 *  that defines the records in the file, the <code>Comparator</code>
 *  and also the <code>BufferedReader</code> and
 *  <code>BufferedWriter</code>.
 *
 *  @see Record
 *  @version 1.0 19.5.97
 *  @author Russel Winder
 */
public interface FileSort
{
    /**
     *  The sort operation.
     *
     *  @param fileName the name of the file to be sorted.
     *
     *  @param r the <code>RecordInformation</code> object that
     *  provides all the information about what a record in the file
     *  looks like.  This includes details of <code>Comparator</code>,
     *  <code>BufferedReader</code> and * <code>BufferedWriter</code>
     */ 
    void sort(String fileName, RecordInformation r)
        throws FileNotFoundException, IOException ;
}

