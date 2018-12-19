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
 *  A function object delivering a balanced merge sort of a file on
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
public final class BalancedMergeSort implements FileSort
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
                               final int blockSize,
                               final int numberOfFiles,
                               final RecordInformation rInfo)
        throws FileNotFoundException, IOException
    {
        //
        //  Create all the files needed for the sorting.
        //
        File file = new File (fileName) ;
        File[] f_A = new File [numberOfFiles] ;
        File[] f_B = new File [numberOfFiles] ;
        for (int i = 0 ; i < numberOfFiles ; i++)
        {
            f_A[i] = new File ("tmp_A_"+i) ;
            f_B[i] = new File ("tmp_B_"+i) ;
        }
        //
        //  Perform the initial dispersion into the A files.
        //
        distributeSortedBlocks(file, f_A, blockSize, rInfo) ;
        //
        //  Undertake the number of merge loops required to guarantee
        //  that everything is sorted.  Remember whether we ended up
        //  with A_0 or B_0 containing the final sorted data.
        //
        File[] from = f_A ;
        File[] to = f_B ;
        boolean B_isFinal = true ;
        for (int i = 0 ;
             merge(from, to, (int)Math.pow(2, i)*blockSize, rInfo) ;
             i++)
        {
            File[] temp = from ;
            from = to ;
            to = temp ;
            B_isFinal = ! B_isFinal ;
        }
        //
        //  Copy the data to the final destination.
        //
        File fileToCopy = B_isFinal ? f_B[0] : f_A[0] ;
        RecordCopyFile.execute(fileToCopy, file, rInfo) ;
        //
        //  Delete all the files.
        //
        for (int i = 0 ; i < numberOfFiles ; i++)
        {
            f_A[i].delete() ;
            f_B[i].delete() ;
        }
    }

    /**
     *  Perform the initial dispersion of the data.
     */    
    private static void distributeSortedBlocks(final File from,
                                               final File[] to,
                                               final int blockSize,
                                        final RecordInformation rInfo)
        throws FileNotFoundException, IOException
    {
        //
        //  Create a Reader for the original data and a set of Writers
        //  for the A files.
        //
        RecordReader reader = rInfo.newRecordReader(from) ;
        RecordWriter[] writers = new RecordWriter[to.length] ;
        for (int i = 0 ; i < to.length ; i++)
        {
            writers[i] = rInfo.newRecordWriter(to[i]) ;
        }
        boolean allDone = false ;
        while(! allDone)
        {
            for (int i = 0 ; ! allDone && (i < writers.length) ; i++)
            {
                //
                //  Pull in a few records, put them into the Vector
                //  that is where we are performing the internal sort
                //  that creates the sorted blocks.
                //
                Vector v = new Vector () ;
                for (int j = 0 ; j < blockSize ; j++)
                {
                    Record r = reader.readRecord() ;
                    if (r == null)
                    {
                    	//System.out.println("I read a null record\n");
                        //
                        //  If we cannot read a record then we have
                        //  reached the end of the file, so we must be
                        //  finished.  Well except that we have to
                        //  sort and write out this incomplete block
                        //  first.
                        //
                        allDone = true ;
                        break ;
                    }
                    /**** my updating ****/
                  //  if (r != null)
                    	/**** end of my updating ****/
                    v.addElement(r) ;
                }
                //
                //  Sort the Vector then write it out to the
                //  appropriate A file.
                //
                QuicksortVector.execute(v, rInfo.getComparator()) ;
                for (int j = 0 ; j < v.size() ; j++)
                {
                    writers[i].writeRecord((Record)v.elementAt(j)) ;
                }
            }
        }
        //
        //  Be tidy and close all the files.  Actually this is
        //  essential to ensure we get a flush.
        //
        for (int i = 0 ; i < writers.length ; i++)
        {
            writers[i].close() ;
        }
        reader.close() ;
    }

    /**
     *  Undertake a round of merging.
     */
    private static boolean merge(final File[] from,
                                 final File[] to,
                                 final int currentBlockSize,
                                 final RecordInformation rInfo)
        throws FileNotFoundException, IOException
    {
        //
        //  Open up the set of Readers and the set of Writers.
        //
        RecordReader[] readers = new RecordReader[from.length] ;
        for (int i = 0 ; i < readers.length ; i++)
        {
            readers[i] = rInfo.newRecordReader(from[i]) ;
        }
        RecordWriter[] writers = new RecordWriter [to.length] ;
        for (int i = 0 ; i < writers.length ; i++)
        {
            writers[i] = rInfo.newRecordWriter(to[i]) ;
        }
        //
        //  We make us of an array which hold the next Record for each
        //  of the files -- we need to have the record in memory in
        //  order to compare the keys and so decide which Record to
        //  write to the output file.
        //
        //  Have another array which is keeping count of how many
        //  Records we take from each of the files so that we can
        //  cease drawing from a given file when we have taken the
        //  appropriate number yet there are still records left.
        //
        boolean returnValue = false ;
        boolean allDone = false ;
        Record[] items = new Record[readers.length] ;
        int[] counts = new int[readers.length] ;
        while (! allDone)
        {
            for (int i = 0 ; i < writers.length ; i++)
            {
                //
                //  Initialize the array holding the next record from
                //  each of the files.  Determine whether we are
                //  finished or not by whether there are any records
                //  left or not.
                //
                allDone = true ;
                for (int j = 0 ; j < readers.length ; j++)
                {
                    counts[j] = 0 ;
                    items[j] = readers[j].readRecord() ;
                    if (items[j] != null)
                    {
                        counts[j] = 1 ;
                        allDone = false ;
                    }
                }
                if (allDone)
                    break ;
                while (true)
                {
                    //
                    //  Determine which is the next Record to add to
                    //  the output stream.  If there isn't one then we
                    //  get a negative index and we can terminate the
                    //  loop.  If we do not terminate then there was a
                    //  Record and we must write it out.
                    //
                    int index = findAppropriate(items,
                                              rInfo.getComparator()) ;
                    if (index < 0)
                        break ;
                    writers[i].writeRecord(items[index]) ;
                    if (i > 0)
                    {
                        //
                        //  We have not yet reduced the problem to
                        //  only a single file so there must be at
                        //  least one more iteration -- we know when
                        //  we are finished when everything goes into
                        //  a single file.
                        //
                        returnValue = true ;
                    }
                    //
                    //  Draw a new Record from the file whose Record
                    //  was chosen -- unless of course we have
                    //  finished our quota from that file.
                    //
                    if (counts[index] < currentBlockSize)
                    {
                        items[index] = readers[index].readRecord() ;
                        if (items[index] != null)
                        {
                            counts[index]++ ;
                        }
                    }
                    else
                    {
                        items[index] = null ;
                    }
                }
            }
        }
        //
        //  Be tidy, close all the files.  Actually this is essentialy
        //  to ensure there is a flush.
        //
        for (int i = 0 ; i < writers.length ; i++)
        {
            writers[i].close() ;
        }
        for (int i = 0 ; i < readers.length ; i++)
        {
            readers[i].close() ;
        }
        return returnValue ;
    }

    /**
     *  Determine which Record is the one to be output next.
     *
     *  @param items the array of <code>Records</code> from which to
     *  select the next according to the order relation defined bu
     *  <code>c</code>.
     *
     *  @param c the <code>Comparator</code> defining the required
     *  order relation on the <code>Record</code>s.
     *
     *  @return the index in the array of the item that should be
     *  chosen next.
     */
    private static int findAppropriate(final Record[] items,
                                       final Comparator c)
    {
        //
        //  Assume no output is to be done and then find the first
        //  non-empty entry.
        //
        int index = -1 ;
        for (int i = 0 ; i < items.length ; i++)
        {
            if (items[i] != null)
            {
                index = i ;
                break ;
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
        return index ;
    }
}

