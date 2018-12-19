/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities.generic_storage;

/**
 *
 * @author raj09hxu
 * @param <T1>
 * @param <T2>
 * Generic Tuple class.
 */
public class Pair <T1, T2>{
    public T1 var1;
    public T2 var2;
    public Pair(T1 t1, T2 t2){
        var1 = t1;
        var2 = t2;
    }
    
    @Override
    public String toString(){
        return var1 + " " + var2;
    }
    
    @Override
    public boolean equals(/*Pair<T1,T2>*/Object ot){

        if (!(ot instanceof Pair)) 
            return false;
        
        Pair<T1, T2> other = (Pair<T1,T2>) ot;
        return var1.equals(other.var1) && var2.equals(other.var2);
    }
}
