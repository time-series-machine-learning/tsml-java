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
 *  An interface defining the necessary properties to be deemed
 *  to be a <code>Comparator</code>.
 *
 *  @version 1.0 2.7.1997
 *  @author Russel Winder
 */
public interface Comparator
{
    /**
     *  The relation that this <code>Comparator</code> represents.
     *
     *  @exception ComparatorParameterErrorException if the type of
     *  the data is not compatible with the type expected by the
     *  <code>Comparator</code>.
     */    
    boolean relation(Object a, Object b) ;
}

