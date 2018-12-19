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
 *  An interface defining the concept of a record writer, the
 *  sort of thing capable of writing a <code>Record</code> to a file.
 *
 *  @see Record
 *  @see RecordReader
 *  @version 1.0 19.5.97
 *  @author Russel Winder
 */
public interface RecordWriter
{
    /**
     *  A <code>Record</code> must be writeable.
     */
    void writeRecord(Record r) throws IOException ;

    /**
     *  A <code>RecordWriter</code> must be closeable.
     */
    void close() throws IOException ;

    /**
     *  A <code>RecordWriter</code> must have a finalizer to clean up
     *  on being garbage collected.
     */
    void finalize() throws IOException ;
}

