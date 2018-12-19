package weka.classifiers.rules.sortinghandler;

/**
 *  This code is from the book:
 *
 *    Winder, R and Roberts, G (1998) <em>Developing Java
 *    Software</em>, John Wiley & Sons.
 *
 *  It is copyright (c) 1997 Russel Winder and Graham Roberts.
 */

import weka.classifiers.rules.sortinghandler.BalancedMergeSort ;

import java.io.FileNotFoundException ;
import java.io.IOException ;
import java.io.Serializable;

public class testFileSort implements Serializable
{
    public static void sort_main (String[] args)
        throws FileNotFoundException, IOException
    {
        System.runFinalizersOnExit(true) ;

        BalancedMergeSort.execute(args[0], Integer.valueOf(args[1]),Integer.valueOf(args[2]), new MyRecordInformation(new MyRecordComparator())) ;
    }
}
