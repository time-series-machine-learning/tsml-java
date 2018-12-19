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
 *  An interface defining the concept of a record reader, the
 *  sort of thing capable of reading a <code>record</code> from a
 *  file.
 *
 *  @see Record
 *  @see RecordWriter
 *  @version 1.0 19.5.97
 *  @author Russel Winder
 */
public interface RecordReader
{
    /**
     *  A <code>Record</code> must be readable.
     */
    Record readRecord() throws IOException ;

    /**
     *  A <code>RecordReader</code> must be closeable.
     */
    void close() throws IOException ;

    /**
     *  A <code>RecordReader</code> must have a finalizer to clean up
     *  on being garbage collected.
     */
    void finalize() throws IOException ;

    /**
     *  Mark an input stream.
     *
     *  @see java.io.BufferedReader#mark
     */
    void mark(int lookAheadLimit)throws IOException ;
    
    /**
     *  Move back to the mark.
     *
     *  @see java.io.BufferedReader#reset
     */
    void reset() throws IOException;
}
