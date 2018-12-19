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
 *  Sort an array of <code>Object</code>s using Quicksort.  This is an
 *  O(n.log(n)) sort except when the data is almost sorted in which
 *  case it O(n^2).
 *
 *  @version 1.0 19.5.97
 *  @author Russel Winder
 */
public class QuicksortVector implements VectorSort
{
    /**
     *  The per object sort operation.
     *
     *  @param v the <code>Vector</code> of <code>Object</code>s to be
     *  sorted.
     *
     *  @param c the <code>Comparator</code> used to compare the
     *  <code>Object</code> during the sort process.
     */ 
    public final void sort(final Vector v, final Comparator c)
    {
        execute(v, c) ;
    }
    
    /**
     *  The statically accessible sort operation.
     *
     *  @param v the <code>Vector</code> of <code>Object</code>s to be
     *  sorted.
     *
     *  @param c the <code>Comparator</code> used to compare the
     *  <code>Object</code> during the sort process.
     */ 
    public static void execute(final Vector v, final Comparator c)
    {
        quicksort(v, 0, v.size()-1, c) ;
    }

    /**
     *  Given the array and two indices, swap the two items in the
     *  array.
     */
    private static void swap(final Vector v,
                             final int a,
                             final int b)
    {
        Object temp = v.elementAt(a) ;
        v.setElementAt(v.elementAt(b), a) ;
        v.setElementAt(temp, b) ;
    }

    /**
     *  Partition an array in two using the pivot value that is at the
     *  centre of the array being partitioned.
     *
     *  <p>This partition implementation based on that in Winder, R
     *  (1993) "Developing C++ Software", Wiley, p.395.  NB. This
     *  implementation (unlike most others) does not guarantee that
     *  the split point contains the pivot value.  Unlike other
     *  implementations, it requires only < (or >) relation and not
     *  both < and <= (or > and >=).  Also, it seems easier to program
     *  and to comprehend.
     *
     *  @param v the array out of which to take a slice.
     *
     *  @param lower the lower bound of this slice.
     *
     *  @param upper the upper bound of this slice.
     *
     *  @param c the <code>Comparator</code> to be used to define the
     *  order.
     */
    private static int partition(final Vector v,
                                       int lower,
                                       int upper,
                                 final Comparator c)
    {
        Object pivotValue = v.elementAt((upper+lower+1)/2) ;
        while (lower <= upper)
        {
            while (c.relation(v.elementAt(lower), pivotValue))
            {
                lower++ ;
            }
            while (c.relation(pivotValue, v.elementAt(upper)))
            {
            	//System.out.println("Lower: "+lower +" "+"upper: "+upper+"\n");
                upper-- ;
            }
            if (lower <= upper)
            {
                if (lower < upper)
                {
                    swap(v, lower, upper) ;
                }
                lower++ ;
                upper-- ;
            }
        }
        return upper ;
    }

    /**
     *  The recursive Quicksort function.
     *
     *  @param v the array out of which to take a slice.
     *
     *  @param lower the lower bound of this slice.
     *
     *  @param upper the upper bound of this slice.
     *
     *  @param c the <code>Comparator</code> to be used to define the
     *  order.
     */
    private static void quicksort(final Vector v,
                                  final int lower,
                                  final int upper,
                                  final Comparator c)
    {
    	//System.out.println("Lower: "+lower +" "+"upper: "+upper+"\n");
        int sliceLength = upper-lower+1 ;
        if (sliceLength > 1)
        {
            if (sliceLength == 2)
            {
                if (c.relation(v.elementAt(upper),v.elementAt(lower)))
                {
                    swap (v, lower, upper) ;
                }
            }
            else
            {
                //
                //  This pivot implementation does not guarantee that
                //  the split point contains the pivot value so we
                //  cannot assume that the pivot is between the two
                //  slices.
                //
                int pivotIndex = partition(v, lower, upper, c) ;
                quicksort(v, lower, pivotIndex, c) ;
                quicksort(v, pivotIndex+1, upper, c) ;
            }
        }
    }
}

