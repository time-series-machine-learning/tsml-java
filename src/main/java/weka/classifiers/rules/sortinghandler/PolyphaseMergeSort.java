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
import java.io.FileWriter ;
import java.io.IOException ;

import java.util.Vector ;

/**
 *  A function object delivering a polyphase merge sort of a file on
 *  the filestore.
 *
 *  @see Record
 *  @see RecordReader
 *  @see RecordWriter
 *  @see RecordInformation
 *  @see RecordCopyFile
 *  @version 1.0 19.5.97
 *  @author Russel Winder
 */
public final class PolyphaseMergeSort implements FileSort
{
    /**
     *  A per object sort operation.
     *
     *  @param fileName the <code>String</code> giving the name of the
     *  file to be sorted.
     *
     *  @param r the <code>RecordInformation</code> factory object for
     *  creating <code>RecordReader</code>s and
     *  <code>RecordWriters</code>s and able to deliver a
     *  <code>Comparator</code>.
     */ 
    public final void sort(final String fileName,
                           final RecordInformation r)
        throws FileNotFoundException, IOException
    {
        execute(fileName, 20, 2, r) ;
    }

    /**
     *  A per object sort operation.
     *
     *  @param fileName the <code>String</code> giving the name of the
     *  file to be sorted.
     *
     *  @param blockSize the number of data items in the initial
     *  sorted blocks.
     *
     *  @param numberOfFiles the number of files to use for initial
     *  dispersion.
     *
     *  @param r the <code>RecordInformation</code> factory object for
     *  creating <code>RecordReader</code>s and
     *  <code>RecordWriters</code>s and able to deliver a
     *  <code>Comparator</code>.
     */ 
    public final void sort(final String fileName,
                           final int blockSize,
                           final int numberOfFiles,
                           final RecordInformation r)
        throws FileNotFoundException, IOException
    {
        execute(fileName, blockSize, numberOfFiles, r) ;
    }

    /**
     *  A statically accessible sort operation.
     *
     *  @param fileName the <code>String</code> giving the name of the
     *  file to be sorted.
     *
     *  @param r the <code>RecordInformation</code> factory object for
     *  creating <code>RecordReader</code>s and
     *  <code>RecordWriters</code>s and able to deliver a
     *  <code>Comparator</code>.
     */ 
    public static void execute(final String fileName,
                               final RecordInformation r)
        throws FileNotFoundException, IOException
    {
        execute(fileName, 20, 2, r) ;
    }
    
    /**
     *  A statically accessible sort operation.
     *
     *  @param fileName the <code>String</code> giving the name of the
     *  file to be sorted.
     *
     *  @param blockSize the number of data items in the initial
     *  sorted blocks.
     *
     *  @param numberOfFiles the number of files to use for initial
     *  dispersion.
     *
     *  @param r the <code>RecordInformation</code> factory object for
     *  creating <code>RecordReader</code>s and
     *  <code>RecordWriters</code>s and able to deliver a
     *  <code>Comparator</code>.
     */ 
    public static void execute(final String fileName,
                               final int approximateBlockSize,
                                     int numberOfFiles,
                               final RecordInformation rInfo)
        throws FileNotFoundException, IOException
    {
        numberOfFiles++ ;
        //
        //  Forceably stick to 3 files.
        //
        numberOfFiles = 3 ;
        //
        //  Create all the files needed for the sorting.
        //
        File file = new File (fileName) ;
        File[] temp = new File [numberOfFiles] ;
        for (int i = 0 ; i < numberOfFiles ; i++)
        {
            temp[i] = new File ("tmp_"+i) ;
        }
        //
        //  Calculate the block size.  Must get things into the
        //  Fibonacci series for it to work properly.
        //
        //  Have to run through the file to find the number of
        //  records.  We need to copy the file anyway so this is not a
        //  wasted activity.
        //
        //  We use the array F to calculate the Fibonacci numbers as
        //  we go, calculating the initialBlockSize to best fit the
        //  nearest Fibonacci number.
        //
        int indexOfNumberOfBlocks = 1 ;
        int initialBlockSize = 1 ;
        int numberOfRecords =
            RecordCopyFile.execute(file, temp[0], rInfo) ;
        int F[] = new int[numberOfRecords] ;
        for (int i = 0 ; i < numberOfFiles ; i++)
        {
            F[i] = 1 ;
        }
        for (int i = numberOfFiles ; i < numberOfRecords ; i++)
        {
            F[i] = 0 ;
            for (int j = i-1 ; j > i-numberOfFiles ; j--)
            {
                F[i] += F[j] ;
            }
            initialBlockSize = numberOfRecords / F[i] ;
            if (initialBlockSize < approximateBlockSize)
            {
                indexOfNumberOfBlocks = i-1 ;
                break ;
            }
        }
        while (true)
        {
            if (++initialBlockSize * F[indexOfNumberOfBlocks] >
                numberOfRecords)
                break ;
        }
        //
        //  Ceate the support arrays containing current block size
        //  and block count in the various files.
        //
        int[] blockSizes = new int[numberOfFiles] ;
        int[] blockCounts = new int[numberOfFiles] ;
        blockSizes[0] = 0 ;
        blockCounts[0] = 0 ;
        for (int i = 1, j = indexOfNumberOfBlocks-1 ;
             i < numberOfFiles ;
             i++, j--)
        {
            blockSizes[i] = initialBlockSize ;
            blockCounts[i] = F[j] ;
        }
        //
        //  Create the files of blocks of sorted records.
        //
        distributeSortedBlocks(temp,
                               0,
                               initialBlockSize,
                               blockCounts,
                               rInfo) ;
        //
        //  Set up the file readers for all the files.
        //
        RecordReader[] readers = new RecordReader[numberOfFiles] ;
        for (int i = 0 ; i < numberOfFiles ; i++)
        {
            readers[i] = rInfo.newRecordReader(temp[i]) ;
        }
        while (true)
        {
            //
            //  Check what work there is to do.  If there is,
            //  find out which is the empty file 
            //
            int toIndex = -1 ;
            int numberOfNonEmptyFiles = 0 ;
            int indexOfNonEmptyFile = -1 ;
            for (int i = 0 ; i < numberOfFiles ; i++)
            {
                if (blockCounts[i] == 0)
                {
                    toIndex = i ;
                }
                else
                {
                    indexOfNonEmptyFile = i ;
                    numberOfNonEmptyFiles++ ;
                }
            }
            //
            //  Exit if everthing is done but close all the files
            //  and copy the result back before exiting.
            //
            if (numberOfNonEmptyFiles <= 1)
            {
                for (int i = 0 ; i < numberOfFiles ; i++)
                {
                    readers[i].close() ;
                }
                RecordCopyFile.execute(temp[indexOfNonEmptyFile],
                                       file,
                                       rInfo) ;
                for (int i = 0 ; i < numberOfFiles ; i++)
                {
                    temp[i].delete() ;
                }
                break ;
            }
            //
            //  Perform the next round of merging.
            //
            readers[toIndex].close() ;
            RecordWriter writer =
                rInfo.newRecordWriter(temp[toIndex]);
            merge(readers,
                  writer,
                  toIndex,
                  blockSizes,
                  blockCounts,
                  rInfo) ;
            writer.close() ;
            readers[toIndex] =
                rInfo.newRecordReader(temp[toIndex]) ;
        }
    }
    
    /**
     *  Perform the initial dispersion of the data.
     */    
    private static void distributeSortedBlocks(File[] files,
                                               int fromIndex,
                                               int blockSize,
                                               int[] blockCounts,
                                              RecordInformation rInfo)
        throws FileNotFoundException, IOException
    {
        //
        //  Create a Reader for the original data and a set of Writers
        //  for the output files.
        //
        RecordReader reader = rInfo.newRecordReader(files[fromIndex]);
        RecordWriter[] writers = new RecordWriter[files.length] ;
        for (int i = 0 ; i < files.length ; i++)
        {
            writers[i] = i == fromIndex
                          ? null
                          : rInfo.newRecordWriter(files[i]) ;
        }
        for (int i = 0 ; i < writers.length ; i++)
        {
            if (i != fromIndex)
            {
                for (int j = 0 ; j < blockCounts[i] ; j++)
                {
                    //
                    //  Pull in a few records, put them into the
                    //  Vector that is where we are performing the
                    //  internal sort that creates us the sorted
                    //  block.
                    //
                    Vector v = new Vector () ;
                    for (int k = 0 ; k < blockSize ; k++)
                    {
                        Record r = reader.readRecord() ;
                        if (r == null)
                            break ;
                        v.addElement(r) ;
                    }
                    //
                    //  Sort the Vector then write it out to the
                    //  appropriate file.
                    //
                    QuicksortVector.execute(v, rInfo.getComparator());
                    for (int k = 0 ; k < v.size() ; k++)
                    {
                       writers[i].writeRecord((Record)v.elementAt(k));
                    }
                }
            }
        }
        //
        //  Be tidy and close all the files.  Actually this is
        //  essential to ensure we get a flush.
        //
        for (int i = 0 ; i < writers.length ; i++)
        {
            if (i != fromIndex)
            {
                writers[i].close() ;
            }
        }
        reader.close() ;
    }

    /**
     *  Undertake a round of merging.
     */
    private static void merge(RecordReader[] readers,
                              RecordWriter writer,
                              int toIndex,
                              int[] blockSizes,
                              int[] blockCounts,
                              RecordInformation rInfo)
        throws FileNotFoundException, IOException
    {
        Record[] items = new Record[readers.length] ;
        int[] counts = new int[readers.length] ;
        int numberOfBlocksMerged = 0 ;
        while (true)
        {
            boolean allDone = false ;
            for (int i = 0 ; i < readers.length ; i++)
            {
                counts[i] = 0 ;
                if (i == toIndex)
                {
                    items[i] = null ;
                }
                else
                {
                    readers[i].mark(64) ;
                    items[i] = readers[i].readRecord() ;
                    if (items[i] == null)
                    {
                        for (int j = 0 ; j < i ; j++)
                        {
                            if (j != toIndex)
                            {
                                readers[j].reset() ;
                            }
                        }
                        allDone = true ;
                        break ;
                    }
                    else
                    {
                        counts[i] = 1 ;
                    }
                }
            }
            if (allDone)
                break ;
            numberOfBlocksMerged++ ;
            while (true)
            {
                int i = findAppropriate(items,
                                        toIndex,
                                        rInfo.getComparator()) ;
                if (i < 0)
                    break ;
                writer.writeRecord(items[i]) ;
                if (counts[i] < blockSizes[i])
                {
                    items[i] = readers[i].readRecord() ;
                    if (items[i] != null)
                    {
                        counts[i]++ ;
                    }
                }
                else
                {
                    items[i] = null ;
                }
            }
        }
        blockSizes[toIndex] = 0 ;
        for (int i = 0 ; i < readers.length ; i++)
        {
            if (i != toIndex)
            {
                blockSizes[toIndex] += blockSizes[i] ;
            }
        }
        for (int i = 0 ; i < readers.length ; i++)
        {
            blockCounts[i] -= numberOfBlocksMerged ;
        }
        blockCounts[toIndex] = numberOfBlocksMerged ;
    }

    /**
     *  Determine which Record is the one to be output next.
     *
     *  @param items the array of <code>Records</code> from which to
     *  select the next according to the order relation defined bu
     *  <code>c</code>.
     *
     *  @param toIndex the index into the array of the target.  The
     *  otheres are assumed to be sources.
     *
     *  @param c the <code>Comparator</code> defining the required
     *  order relation on the <code>Record</code>s.
     *
     *  @return the index in the array of the item that should be
     *  chosen next.
     */
    private static int findAppropriate(Record[] items,
                                       int toIndex,
                                       Comparator c)
    {
        //
        //  Assume no output is to be done and then find the first
        //  non-empty entry.
        //
        int index = -1 ;
        for (int i = 0 ; i < items.length ; i++)
        {
            if (i != toIndex)
            {
                if (items[i] != null)
                {
                    index = i ;
                    break ;
                }
            }
        }
        //
        //  If there were no non-empty entries then do nothing, we are
        //  finshied.  Otherwise...
        //
        if (index >= 0) 
        {
            //
            //  ...do a linear search through the items to see which
            //  is the next one to select.
            //
            Record value = items[index] ;
            for (int i = index+1 ; i < items.length ; i++)
            {
                if (i != toIndex)
                {
                    if (items[i] != null)
                    {
                        if (c.relation(items[i], value))
                        {
                            index = i ;
                            value = items[i] ;
                        }
                    }
                }
            }
        }
        return index ;
    }
}
