package weka.classifiers.rules.sortinghandler;

/**
 *  This code is from the book:
 *
 *    Winder, R and Roberts, G (1998) <em>Developing Java
 *    Software</em>, John Wiley & Sons.
 *
 *  It is copyright (c) 1997 Russel Winder and Graham Roberts.
 */


/**
 *  An interface defining the concept of a record, the sort of thing
 *  written to and read from a file.
 *
 *  @see RecordReader
 *  @see RecordWriter
 *  @version 1.0 19.5.97
 *  @author Russel Winder
 */
public interface Record
{
    /**
     *  A <code>Record</code> must have a key that the records can be
     *  ordered on.
     */
    int key() ;

    /**
     *  A <code>Record</code> must have a printed form.
     */
    String toString() ;
}
