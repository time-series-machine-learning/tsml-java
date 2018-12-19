package weka.classifiers.rules.sortinghandler;

/**
 *  This code is from the book:
 *
 *    Winder, R and Roberts, G (1998) <em>Developing Java
 *    Software</em>, John Wiley & Sons.
 *
 *  It is copyright (c) 1997 Russel Winder and Graham Roberts.
 */


import java.util.Vector ;

/**
 *  The interface implemented by any <code>Object</code> array
 *  sorting function object.
 *
 *  @version 1.0 19.5.97
 *  @author Russel Winder
 */
public interface VectorSort
{
    /**
     *  The sort operation.
     *
     *  @param v the <code>Vector</code> of <code>Object</code>s to be
     *  sorted.
     *
     *  @param c the <code>Comparator</code> used to compare the
     *  <code>Object</code> during the sort process.
     */ 
    void sort(Vector v, Comparator c) ;
}

