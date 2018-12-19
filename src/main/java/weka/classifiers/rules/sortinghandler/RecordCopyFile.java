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
import java.io.FileNotFoundException ;
import java.io.IOException ;

/**
 *  A function object to copy a file containing <code>Record</code>s..
 *
 *  @see Record
 *  @see RecordReader
 *  @see RecordWriter
 *  @see RecordInformation
 *  @see BalancedMergeSort
 *  @see PolyPhaseMergeSort
 *  @version 1.0 2.7.1997
 *  @author Russel Winder
 */
public final class RecordCopyFile
{
    /**
     *  The per object copy function.
     *
     *  @param from the file to read <code>Record</code>s from.
     *
     *  @param to the file to write <code>Record</code>s from.
     *
     *  @param rInfo the factory object required to be able to
     *  construct the <code>Reader</code> and <code>Writer</code> of
     *  <code>Record</code>s.
     *
     *  @return the number of records copied.
     */    
    public int copy(final File from,
                    final File to,
                    final RecordInformation rInfo)
        throws FileNotFoundException, IOException
    {
        return execute(from, to, rInfo) ;
    }

    /**
     *  The static access to the copying function object.
     *
     *  @param from the file to read <code>Record</code>s from.
     *
     *  @param to the file to write <code>Record</code>s from.
     *
     *  @param rInfo the factory object required to be able to
     *  construct the <code>Reader</code> and <code>Writer</code> of
     *  <code>Record</code>s.
     *
     *  @return the number of records copied.
     */
    public static int execute(final File from,
                              final File to,
                              final RecordInformation rInfo)
        throws FileNotFoundException, IOException
    {
        //
        //  Set up the Reader and the Writer.
        //
        RecordReader source = rInfo.newRecordReader(from) ;
        RecordWriter target = rInfo.newRecordWriter(to) ;
        //
        //  Copy all the Records from the Reader to the Writer.
        //
        int count = 0 ;
        while (true)
        {
            Record r = source.readRecord() ;
            if (r == null)
                break ;
            target.writeRecord(r) ;
            count++ ;
        }
        //
        //  Close the files and ensure the flush.
        //
        source.close() ;
        target.close() ;
        return count ;
    }
}

