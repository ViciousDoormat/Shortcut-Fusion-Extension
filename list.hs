--requires running cabal install --lib containers

{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE UndecidableInstances #-}

import Data.Sequence

main :: IO ()
main = do 
    print $ maximum' $ reverse' $ filter' even (Two 1 $ Two 2 $ Two 3 $ Two 4 $ Two 5 No)
    print $ filter' even $ append' (Two 1 $ Two 2 $ Two 3 $ Two 4 $ Two 5 No) (between (6,10))
    print $ toSeq $ filter' even $ append' (Two 1 $ Two 2 $ Two 3 $ Two 4 $ Two 5 No) (between (6,10))
    print $ toSeq $ reverse' $ between (1,5)
    print $ filter' even $ map' (+1) $ filter' even $ between (1,5)

--The main structure, its base functor, and church encoding
--These parts can be defined without any problems
data List a = No | One a | Two a (List a) deriving Show
data List_ a b = No_ | One_ a | Two_ a b 
--Note that the Church encoding requires Usable a b
newtype ListCh a = ListCh (forall b. Usable a b => (List_ a b -> b) -> b)

fold :: (List_ a b -> b) -> List a -> b
fold a No = a No_
fold a (One x) = a (One_ x)
fold a (Two x xs) = a (Two_ x (fold a xs))

toCh :: List a -> ListCh a
toCh xs = ListCh (`fold` xs)
{-# INLINE [0] toCh #-}

in' :: List_ a (List a) -> List a
in' No_ = No
in' (One_ x) = One x
in' (Two_ x xs) = Two x xs

fromCh :: ListCh a -> List a
fromCh (ListCh g) = g in'
{-# INLINE [0] fromCh #-}

{-# RULES "toCh/fromCh fusion" forall x. toCh (fromCh x) = x #-}

--between can be defined without any trouble
betweenCh :: (Int,Int) -> ListCh Int
betweenCh (x,y) = ListCh (`loop` (x,y))
  where
    loop a (x,y)
      | x > y  = a No_
      | x == y = a (One_ x)
      | x < y  = a (Two_ x (loop a (x+1,y)))

between :: (Int,Int) -> List Int
between = fromCh . betweenCh
{-# INLINE between #-}

--sum can be defined without any trouble
s :: List_ Int Int -> Int
s No_ = 0
s (One_ x) = x
s (Two_ x y) = x + y

sumCh :: ListCh Int -> Int
sumCh (ListCh g) = g s

sum' :: List Int -> Int
sum' = sumCh . toCh
{-# INLINE sum' #-}

--maximum can be defined without any trouble
mx :: List_ Int Int -> Int
mx No_ = minBound
mx (One_ x) = x
mx (Two_ x y) = max x y

maximumCh :: ListCh Int -> Int
maximumCh (ListCh g) = g mx

maximum' :: List Int -> Int
maximum' = maximumCh . toCh
{-# INLINE maximum' #-}


--The implementations of the transformers:

--We need a way to potentially filter the x away but do return a List_ a b. 
--We get stuck if we do (if f x then Two_ x xs else ...) because at the ... we cannot generally remove the x and return a List_ a b, using only the remaining c.
--With the Tree from the paper, this could be done, because there is no branch that holds a value. Only the end points (Leaf a) hold values,
--and these can be filtered to Empty. 
--With the Usable constraint, we are guaranteed that b can form a List_ on its own with oneStepFilter.
f :: Usable a b => (a -> Bool) -> List_ a b -> List_ a b
f p No_ = No_
f p (One_ x) = if p x then One_ x else No_
f p (Two_ x xs) = oneStepFilter p x xs 

filterCh :: (a -> Bool) -> ListCh a -> ListCh a
filterCh p (ListCh g) = ListCh (\a -> g (a . f p))

filter' :: (a -> Bool) -> List a -> List a
filter' p = fromCh . filterCh p . toCh
{-# INLINE filter' #-}

--Here, we require the use of oneStepAppend, because we cannot generally create a List_ from two "branches".
--The List_ structure does not have a constructor Binary b b, which makes this possible in the Tree structure from the paper.
--With the Usable constraint, we know that these two "end points" can be appended into one with oneStepAppend.
appendCh :: ListCh a -> ListCh a -> ListCh a
appendCh (ListCh g1) (ListCh g2) = ListCh (\a -> a (oneStepAppend (g1 a) (g2 a)))

append' :: List a -> List a -> List a
append' xs ys = fromCh (appendCh (toCh xs) (toCh ys))
{-# INLINE append' #-}

--Here, we require the use of oneStepReverse, because we cannot generally put an x::a "at the end" of an xs::c.
--In the paper, this could be done because the Tree sructure does not encode a linear datastructure.
--Reversing for a Tree does only require to swich the branches. No values have to be moved "inwards".
--With the Usable constraint, we know that x::a and xs::c are such that this can be done with oneStepReverse.
r :: Usable a c => List_ a c -> List_ a c
r No_ = No_
r (One_ x) = One_ x
r (Two_ x xs) = oneStepReverse x xs

reverseCh :: ListCh a -> ListCh a
reverseCh (ListCh g) = ListCh (\a -> g (a . r))

reverse' :: List a -> List a
reverse' = fromCh . reverseCh . toCh
{-# INLINE reverse' #-}


--The case class and its instances required to implement transformation functions:

--Gives the guarantee that a can be merged with b in the desired ways,
--such that the merger describes one step of the recursive process.
class Usable a b where
  oneStepFilter :: (a -> Bool) -> a -> b -> List_ a b
  oneStepAppend :: b -> b -> List_ a b
  oneStepReverse :: a -> b -> List_ a b

--Because we work with ListCh Int (because of sum and maximum), we are required to define the following instance.
--All these functions actually are one step non-recursive implementations.
instance Usable b b where
  --Here, we require the One_. We cannot make a Two_ with two b's if we only have one b which we need to use once.
  oneStepFilter :: (b -> Bool) -> b -> b -> List_ b b
  oneStepFilter f x y = if f x then Two_ x y else One_ y

  oneStepAppend :: b -> b -> List_ b b
  oneStepAppend = Two_

  oneStepReverse :: b -> b -> List_ b b
  oneStepReverse x y = Two_ y x

--Because we work with ListCh (Tree a) (because of fromCh), we are required to define the following instance.
instance Usable a (List a) where
  oneStepFilter :: (a -> Bool) -> a -> List a -> List_ a (List a)
  oneStepFilter f v No = if f v then One_ v else No_
  oneStepFilter f v (One x) = if f v then Two_ v (One x) else One_ x
  oneStepFilter f v (Two x xs) = if f v then Two_ v (Two x xs) else Two_ x xs

  --We cannot put an x::a at the end of a ys::(List a) without going through all of ys.
  --Therefore we cannot define one step of the reverse function for (Usable a (List a)) non-recursively.
  oneStepReverse :: a -> List a -> List_ a (List a)
  oneStepReverse x No = One_ x
  oneStepReverse x (One y) = Two_ y (One x)
  --We need to call oneStepReverse again.
  oneStepReverse x (Two y ys) = Two_ y (in' (oneStepReverse x ys))

  --This function too cannot be made non-recursively.
  oneStepAppend :: List a -> List a -> List_ a (List a)
  oneStepAppend No No = No_
  oneStepAppend No (One y) = One_ y
  oneStepAppend (One x) No = One_ x
  oneStepAppend No (Two y ys) = Two_ y ys
  oneStepAppend (Two x xs) No = Two_ x xs
  oneStepAppend (One x) ys = Two_ x ys
  --We need to call oneStepAppend again.
  oneStepAppend (Two x xs) ys = Two_ x (in' (oneStepAppend xs ys))

--I use Sequence instead of a Haskell List, because sequences allow O(1) appending an element at the end, instead of the O(n) xs ++ [x],
--and O(log(min(n1,n2))) append instead of xs ++ ys.
--This makes it posible to implement a toSeq function that makes printing a tree nice, whilst allowing great/decent performance improvements.
instance Usable a (Seq a) where
  --In contrast to the Usable a (List a) instance, this implementation of oneStepReverse is O(1)
  oneStepReverse :: a -> Seq a -> List_ a (Seq a)
  oneStepReverse x Empty = One_ x
  oneStepReverse x (y:<|xs) = Two_ y (xs |> x)
  
  --In contrast to the Usable a (List a) instance, this implementation of oneStepReverse is O(log(min(n1,n2))).
  oneStepAppend :: Seq a -> Seq a -> List_ a (Seq a)
  oneStepAppend (x:<|xs) ys = Two_ x (xs >< ys)
  
  oneStepFilter :: (a -> Bool) -> a -> Seq a -> List_ a (Seq a)
  oneStepFilter f v Empty = if f v then One_ v else No_
  oneStepFilter f v (x :<| Empty) = if f v then Two_ v (x :<| Empty) else One_ x
  oneStepFilter f v (x :<| xs) = if f v then Two_ v (x :<| xs) else Two_ x xs

--toSeq can be defined without any trouble, similarly to the consumers sum and maximum.
ts :: List_ a (Seq a) -> Seq a
ts No_ = Empty
ts (One_ x) = singleton x
ts (Two_ x xs) = x <| xs

toSeqCh :: ListCh a -> Seq a
toSeqCh (ListCh g) = g ts

toSeq :: List a -> Seq a
toSeq = toSeqCh . toCh
{-# INLINE toSeq #-}


--map could in theory be defined without any trouble,
--as it does not suffer from the problems of the other transformation functions.
--However, now that we made our implementations with the Usable constraints, map has a severe limitation:
--(k::List c b -> b) needs to be restricted such that Usable a b, because g requires this restriction.
--g has the guarantee for Usable a b, but not for Usable c b. Thus we must know that if we have Usable c b, we also have Usable a b
--In other words, that the type of the value that k creates will still be Usable with the value a.
--Therefore, we need to give Haskell this guarantee ourselves. The most useful thing I could think of was the constraint
--(forall b. Usable c b => Usable a b). However, this poses a big limitation to the system.
--A non-trivial case of this implication is when the mapping actually changes the type (e.g. even::Int->Bool).
--In that case, Haskell cannot figure out the implication on its own.
--However, I have no clue how to prove this implication for these non-trivial cases myself.
--This means that map can only be used in cases where a == c (e.g. (+1)::Int->Int).
mapCh :: (forall b. Usable c b => Usable a b) => (a -> c) -> ListCh a -> ListCh c
mapCh f (ListCh g) = ListCh (\k -> g (k . m f)) 

m :: (a -> b) -> List_ a c -> List_ b c
m f No_ = No_
m f (One_ x) = One_ $ f x
m f (Two_ x xs) = Two_ (f x) xs 

map' :: (forall b. Usable c b => Usable a b) => (a -> c) -> List a -> List c
map' f = fromCh . mapCh f . toCh
{-# INLINE map' #-}