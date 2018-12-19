/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities;

import java.util.Objects;
import java.util.function.Function;

public interface TriFunction<A,B,C,R> {

    R apply(A a, B b, C c);

    default <V> TriFunction<A, B, C, V> andThen(
                                Function<? super R, ? extends V> after) {
        Objects.requireNonNull(after);
        return (A a, B b, C c) -> after.apply(apply(a, b, c));
    }
}
