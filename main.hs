--requires running cabal install --lib criterion
--requires running cabal install --lib deepseq
--needs to be compiled with the O flag; ghc -O main.hs

{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

import GHC.Generics (Generic, Generic1)
import Control.DeepSeq --To force the structures to normal form and to actually be calculated.
import Criterion.Main --For benchmarking.

--For all benchmarks except append pipeline fused: compile s.t. RULES toCh/fromCh and toCoCh/fromCoCh ENABLED + all INLINE statements ENABLED.
--Nothing needs to be done for this. The file as it is now is according to this requirement.
--The last benchmark (append pipeline) contains special compilation instructions explained below.
main = defaultMain [
  --Functions on their own:
  bgroup "sum"     [ bench "normal"   $ nf sum_  (between (1,10000))
                   , bench "chirch"   $ nf sum'  (between (1,10000))
                   , bench "cochirch" $ nf sum'' (between (1,10000))
                   ],
  bgroup "maximum" [ bench "normal"   $ nf maximum_  (between (1,10000))
                   , bench "chirch"   $ nf maximum'  (between (1,10000))
                   , bench "cochirch" $ nf maximum'' (between (1,10000))
                   ],
  bgroup "between" [ bench "normal"   $ nf between   (1,10000)
                   , bench "chirch"   $ nf between'  (1,10000)
                   , bench "cochirch" $ nf between'' (1,10000)
                   ],
  bgroup "map"     [ bench "normal"   $ nf (map_  (+1)) (between (1,10000))
                   , bench "chirch"   $ nf (map'  (+1)) (between (1,10000))
                   , bench "cochirch" $ nf (map'' (+1)) (between (1,10000))
                   ],
  bgroup "reverse" [ bench "normal"   $ nf reverse_  (between (1,10000))
                   , bench "chirch"   $ nf reverse'  (between (1,10000))
                   , bench "cochirch" $ nf reverse'' (between (1,10000))
                   ],
  bgroup "filter"  [ bench "normal"   $ nf (filter_  odd) (between (1,10000))
                   , bench "chirch"   $ nf (filter'  odd) (between (1,10000))
                   , bench "cochirch" $ nf (filter'' odd) (between (1,10000))
                   ],
  bgroup "append"  [ bench "normal"   $ nf (uncurry append)   (between (1,10000), between (1,10000))
                   , bench "chirch"   $ nf (uncurry append')  (between (1,10000), between (1,10000))
                   , bench "cochirch" $ nf (uncurry append'') (between (1,10000), between (1,10000))
                   ],

  --Small pipelines:
  bgroup "max.bet" [ bench "normal"   $ nf (maximum_  . between)   (1,10000)
                   , bench "chirch"   $ nf (maximum'  . between')  (1,10000)
                   , bench "cochirch" $ nf (maximum'' . between'') (1,10000)
                   ],
  bgroup "max.fit" [ bench "normal"   $ nf (maximum_  . filter_  odd) (between (1,10000))
                   , bench "chirch"   $ nf (maximum'  . filter'  odd) (between (1,10000))
                   , bench "cochirch" $ nf (maximum'' . filter'' odd) (between (1,10000))
                   ],
  bgroup "max.map" [ bench "normal"   $ nf (maximum_  . map_  (+1)) (between (1,10000))
                   , bench "chirch"   $ nf (maximum'  . map'  (+1)) (between (1,10000))
                   , bench "cochirch" $ nf (maximum'' . map'' (+1)) (between (1,10000))
                   ],
  bgroup "sum.bet" [ bench "normal"   $ nf (sum_  . between)   (1,10000)
                   , bench "chirch"   $ nf (sum'  . between')  (1,10000)
                   , bench "cochirch" $ nf (sum'' . between'') (1,10000)
                   ],
  bgroup "sum.fit" [ bench "normal"   $ nf (sum_  . filter_  odd) (between (1,10000))
                   , bench "chirch"   $ nf (sum'  . filter'  odd) (between (1,10000))
                   , bench "cochirch" $ nf (sum'' . filter'' odd) (between (1,10000))
                   ],
  bgroup "sum.map" [ bench "normal"   $ nf (sum_  . map_  (+1)) (between (1,10000))
                   , bench "chirch"   $ nf (sum'  . map'  (+1)) (between (1,10000))
                   , bench "cochirch" $ nf (sum'' . map'' (+1)) (between (1,10000))
                   ],
  bgroup "rev.map" [ bench "normal"   $ nf (reverse_  . map_  (+1)) (between (1,10000))
                   , bench "chirch"   $ nf (reverse'  . map'  (+1)) (between (1,10000))
                   , bench "cochirch" $ nf (reverse'' . map'' (+1)) (between (1,10000))
                   ],
  bgroup "map.rev" [ bench "normal"   $ nf (map_  (+1) . reverse_)  (between (1,10000))
                   , bench "chirch"   $ nf (map'  (+1) . reverse')  (between (1,10000))
                   , bench "cochirch" $ nf (map'' (+1) . reverse'') (between (1,10000))
                   ],
  bgroup "rev.fit" [ bench "normal"   $ nf (reverse_  . filter_  odd) (between (1,10000))
                   , bench "chirch"   $ nf (reverse'  . filter'  odd) (between (1,10000))
                   , bench "cochirch" $ nf (reverse'' . filter'' odd) (between (1,10000))
                   ],
  bgroup "fit.rev" [ bench "normal"   $ nf (filter_  odd . reverse_)  (between (1,10000))
                   , bench "chirch"   $ nf (filter'  odd . reverse')  (between (1,10000))
                   , bench "cochirch" $ nf (filter'' odd . reverse'') (between (1,10000))
                   ],
  bgroup "sum.rev" [ bench "normal"   $ nf (sum_  . reverse_)  (between (1,10000))
                   , bench "chirch"   $ nf (sum'  . reverse')  (between (1,10000))
                   , bench "cochirch" $ nf (sum'' . reverse'') (between (1,10000))
                   ],
  bgroup "max.rev" [ bench "normal"   $ nf (maximum_  . reverse_)  (between (1,10000))
                   , bench "chirch"   $ nf (maximum'  . reverse')  (between (1,10000))
                   , bench "cochirch" $ nf (maximum'' . reverse'') (between (1,10000))
                   ],
  bgroup "rev.bet" [ bench "normal"   $ nf (reverse_  . between)   (1,10000)
                   , bench "chirch"   $ nf (reverse'  . between')  (1,10000)
                   , bench "cochirch" $ nf (reverse'' . between'') (1,10000)
                   ],
  
  --large pipelines:
  bgroup "pipeline" [ bench "normal"   $ nf (sum_  . map_  (+1) . filter_  odd . between)   (1,10000)
                    , bench "chirch"   $ nf (sum'  . map'  (+1) . filter'  odd . between')  (1,10000)
                    , bench "cochirch" $ nf (sum'' . map'' (+1) . filter'' odd . between'') (1,10000)
                    ],

  --append pipeline needs to be executed two times.
  --append pipeline fused: compiled s.t. RULES toCh/fromCh and toCoCh/fromCoCh ENABLED + all INLINE statements ENABLED    (the file as it currently is)
  --append pipeline unfused: compiled s.t. RULES toCh/fromCh and toCoCh/fromCoCh DISABLED + all INLINE statements DISABLED
  bgroup "append pl" [ bench "normal"   $ nf sumApp    (1,10000)
                     , bench "chirch"   $ nf sumApp'   (1,10000)
                     , bench "cochirch" $ nf sumApp'' (1,10000)
                     ]  
  ]

--The main datastructure used in this experiment.
--The deriving statements make sure that during benchmarking, the structure gets fully executed until normal form.
--Otherwise, due to Haskell's lazyness, all benchmark results will execute until WHNF and take 30 microseconds.
data Tree a = Empty | Leaf a | Fork (Tree a) (Tree a) deriving (Generic, NFData, Show)

--The base functor of Tree used by the Church and CoChurch encodings.
data Tree_ a b = Empty_ | Leaf_ a | Fork_ b b


--Below, all standard versions of the functions (defined over Tree) and some one step versions can be found, as defined/described in the paper:

--Standard between function over Tree.
between :: (Int, Int) -> Tree Int
between (x,y)
  | x > y  = Empty
  | x == y = Leaf x
  | x < y  = Fork (between (x,mid)) (between (mid+1,y))
  where
    mid = (x+y) `div` 2

--Standard sum function over Tree.
sum_ :: Tree Int -> Int
sum_ Empty      = 0
sum_ (Leaf x)   = x
sum_ (Fork l r) = sum_ l + sum_ r

--Standard reverse function over Tree.
reverse_ :: Tree a -> Tree a
reverse_ Empty = Empty
reverse_ (Leaf a) = Leaf a
reverse_ (Fork l r) = Fork (reverse_ r) (reverse_ l)

--One step reverse. Used by the Church and CoChurch versions of reverse.
r :: Tree_ a c -> Tree_ a c
r Empty_ = Empty_
r (Leaf_ a) = Leaf_ a
r (Fork_ l r) = Fork_ r l

--Standard filter function over Tree.
filter_ :: (a -> Bool) -> Tree a -> Tree a
filter_ p Empty      = Empty
filter_ p (Leaf a)   = if p a then Leaf a else Empty
filter_ p (Fork l r) = append (filter_ p l) (filter_ p r)

--One step filter. Used by the Church and CoChurch versions of filter.
f :: (a -> Bool) -> Tree_ a c -> Tree_ a c
f p Empty_ = Empty_
f p (Leaf_ x) = if p x then Leaf_ x else Empty_
f p (Fork_ l r) = Fork_ l r

--Standard append function over Tree.
append :: Tree a -> Tree a -> Tree a
append t1 Empty = t1
append Empty t2 = t2
append t1 t2 = Fork t1 t2

--Standard map function over Tree.
map_ :: (a -> b) -> Tree a -> Tree b
map_ f Empty = Empty
map_ f (Leaf a) = Leaf (f a)
map_ f (Fork l r) = Fork (map_ f l) (map_ f r)

--One step map. Used by the Church and CoChurch versions of map.
m :: (a -> b) -> Tree_ a c -> Tree_ b c
m f Empty_ = Empty_
m f  (Leaf_ x) = Leaf_ (f x)
m f (Fork_ l r) = Fork_ l r

--Standard maximum function over Tree.
maximum_ :: Tree Int -> Int
maximum_ Empty = minBound
maximum_ (Leaf x) = x
maximum_ (Fork l r) = let max1 = maximum_ l; max2 = maximum_ r in max max1 max2

sumApp :: (Int, Int) -> Int
sumApp (x,y) = sum_ $ append (between (x,y)) (between (x,y))

--Below, the Church encoding and its versions of the functions can be found, as defined/described in the paper:

--The Church encoding of Tree.
newtype TreeCh a = TreeCh (forall b. (Tree_ a b -> b) -> b)

fold :: (Tree_ a b -> b) -> Tree a -> b
fold a Empty      = a Empty_
fold a (Leaf x)   = a (Leaf_ x)
fold a (Fork l r) = a (Fork_ (fold a l) (fold a r))

toCh :: Tree a -> TreeCh a
toCh t = TreeCh (`fold` t)
{-# INLINE [0] toCh #-}

in' :: Tree_ a (Tree a) -> Tree a
in' Empty_ = Empty 
in' (Leaf_ x) = Leaf x
in' (Fork_ l r) = Fork l r

fromCh :: TreeCh a -> Tree a
fromCh (TreeCh fold) = fold in'
{-# INLINE [0] fromCh #-}

{-# RULES "toCh/fromCh fusion" forall x. toCh (fromCh x) = x #-}

betweenCh :: (Int,Int) -> TreeCh Int
betweenCh (x,y) = TreeCh (`loop` (x,y))
  where
    loop a (x,y)
      | x > y  = a Empty_
      | x == y = a (Leaf_ x)
      | x < y  = a (Fork_ (loop a (x,mid)) (loop a (mid+1,y)))
      where 
        mid = (x+y) `div` 2

--Church encoded between function over Tree.
between' :: (Int,Int) -> Tree Int
between' = fromCh . betweenCh
{-# INLINE between' #-}

--One step sum.
s :: Tree_ Int Int -> Int
s Empty_      = 0
s (Leaf_ x)   = x
s (Fork_ x y) = x + y

sumCh :: TreeCh Int -> Int
sumCh (TreeCh g) = g s

--Church encoded sum function over Tree.
sum' :: Tree Int -> Int
sum' = sumCh . toCh
{-# INLINE sum' #-}

--One step maximum.
mx :: Tree_ Int Int -> Int
mx Empty_ = minBound
mx (Leaf_ x)   = x
mx (Fork_ x y) = max x y

maximumCh :: TreeCh Int -> Int
maximumCh (TreeCh g) = g mx

--Church encoded maximum function over Tree.
maximum' :: Tree Int -> Int
maximum' = maximumCh . toCh
{-# INLINE maximum' #-}

reverseCh :: TreeCh a -> TreeCh a
reverseCh (TreeCh g) = TreeCh (\a -> g (a . r))

--Church encoded reverse function over Tree.
reverse' :: Tree a -> Tree a
reverse' = fromCh . reverseCh . toCh
{-# INLINE reverse' #-}

filterCh :: (a -> Bool) -> TreeCh a -> TreeCh a
filterCh p (TreeCh g) = TreeCh (\a -> g (a . f p))

--Church encoded reverse function over Tree.
filter' :: (a -> Bool) -> Tree a -> Tree a
filter' p = fromCh . filterCh p . toCh
{-# INLINE filter' #-}

appendCh :: TreeCh a -> TreeCh a -> TreeCh a
appendCh (TreeCh g1) (TreeCh g2) = 
    TreeCh (\a -> a (Fork_ (g1 a) (g2 a)))

--Church encoded append function over Tree.
append' :: Tree a -> Tree a -> Tree a
append' t1 t2 = fromCh (appendCh (toCh t1) (toCh t2))
{-# INLINE append' #-}

mapCh :: (a -> b) -> TreeCh a -> TreeCh b
mapCh f (TreeCh g) = TreeCh (\a -> g (a . m f))

--Church encoded map function over Tree.
map' :: (a -> b) -> Tree a -> Tree b
map' f = fromCh. mapCh f . toCh
{-# INLINE map' #-}

sumApp' :: (Int, Int) -> Int
sumApp' (x,y) = sum' $ append' (between' (x,y)) (between' (x,y))


--These rules make it possible for Haskell to switch from a Church encoded version of append to the normal version, and vice versa.
--The normal version is faster over very small pipelines, in which case we would like to make said switch.
--As these two rules are not in used and interfer with the benchmarking, they are commented out.
-- {-# RULES
-- "append -> fusedCh" [~1] forall t1 t2.
--   append t1 t2 =
--     fromCh (appendCh (toCh t1) (toCh t2))

-- "append -> unfusedCh" [1] forall t1 t2.
--   fromCh (appendCh (toCh t1) (toCh t2)) =
--     append t1 t2
-- #-}


--Below, the CoChurch encoding and its versions of the functions can be found, as defined/described in the paper:

--The CoChurch encoding of Tree.
data TreeCoCh a = forall s. TreeCoCh (s -> Tree_ a s) s

out :: Tree a -> Tree_ a (Tree a)
out Empty = Empty_
out (Leaf a) = Leaf_ a
out (Fork l r) = Fork_ l r

toCoCh :: Tree a -> TreeCoCh a
toCoCh = TreeCoCh out
{-# INLINE [0] toCoCh #-}

unfold :: (s -> Tree_ a s) -> s -> Tree a
unfold h s = case h s of
  Empty_ -> Empty
  Leaf_ a -> Leaf a
  Fork_ sl sr -> Fork (unfold h sl) (unfold h sr)

fromCoCh :: TreeCoCh a -> Tree a
fromCoCh (TreeCoCh h s) = unfold h s
{-# INLINE [0] fromCoCh #-}

{-# RULES "toCoCh/fromCoCh fusion" forall x. toCoCh (fromCoCh x) = x #-}

betweenCoCh :: (Int,Int) -> TreeCoCh Int
betweenCoCh (x,y) = TreeCoCh h (x,y)
  where
    h (x,y) | x > y  = Empty_
            | x == y = Leaf_ x
            | x < y  = Fork_ (x,mid) (mid+1,y)
            where
              mid = (x+y) `div` 2

--CoChurch encoded between function over Tree.
between'' :: (Int,Int) -> Tree Int
between'' = fromCoCh . betweenCoCh
{-# INLINE between'' #-}

sumCoCh :: TreeCoCh Int -> Int
sumCoCh (TreeCoCh h s) = loop s
  where
    loop s = case h s of
      Empty_    -> 0
      Leaf_ x   -> x
      Fork_ l r -> loop l + loop r

--CoChurch encoded sum function over Tree.
sum'' :: Tree Int -> Int
sum'' = sumCoCh . toCoCh
{-# INLINE sum'' #-}

maximumCoCh :: TreeCoCh Int -> Int
maximumCoCh (TreeCoCh h s) = loop s
  where
    loop s = case h s of
      Empty_    -> minBound
      Leaf_ x   -> x
      Fork_ l r -> let max1 = loop l; max2 = loop r in max max1 max2

--CoChurch encoded maximum function over Tree.
maximum'' :: Tree Int -> Int
maximum'' = maximumCoCh . toCoCh
{-# INLINE maximum'' #-}

reverseCoCh :: TreeCoCh a -> TreeCoCh a
reverseCoCh (TreeCoCh h s) = TreeCoCh (r . h) s

--CoChurch encoded reverse function over Tree.
reverse'' :: Tree a -> Tree a
reverse'' = fromCoCh . reverseCoCh . toCoCh
{-# INLINE reverse'' #-}

filterCoCh :: (a -> Bool) -> TreeCoCh a -> TreeCoCh a
filterCoCh p (TreeCoCh h s) = TreeCoCh (f p . h) s

--CoChurch encoded filter function over Tree.
filter'' :: (a -> Bool) -> Tree a -> Tree a
filter'' p = fromCoCh . filterCoCh p . toCoCh
{-# INLINE filter'' #-}

appendCoCh :: TreeCoCh a -> TreeCoCh a -> TreeCoCh a
appendCoCh (TreeCoCh h1 s1) (TreeCoCh h2 s2) = TreeCoCh h' Nothing
  where
    h' Nothing = Fork_ (Just (TreeCoCh h1 s1)) (Just (TreeCoCh h2 s2))
    h' (Just (TreeCoCh h s)) = case h s of
      Empty_ -> Empty_
      Leaf_ a -> Leaf_ a
      Fork_ l r -> Fork_ (Just (TreeCoCh h l)) (Just (TreeCoCh h r))

--CoChurch encoded append function over Tree.
append'' :: Tree a -> Tree a -> Tree a
append'' t1 t2 = fromCoCh (appendCoCh (toCoCh t1) (toCoCh t2))
{-# INLINE append'' #-}

mapCoCh :: (a -> b) -> TreeCoCh a -> TreeCoCh b
mapCoCh f (TreeCoCh h s) = TreeCoCh (m f . h) s

--CoChurch encoded map function over Tree.
map'' :: (a -> b) -> Tree a -> Tree b
map'' f = fromCoCh . mapCoCh f . toCoCh
{-# INLINE map'' #-}

sumApp'' :: (Int, Int) -> Int
sumApp'' (x,y) = sum'' $ append'' (between'' (x,y)) (between'' (x,y))

--These rules make it possible for Haskell to switch from a CoChurch encoded version of append to the normal version, and vice versa.
--The normal version is faster over very small pipelines, in which case we would like to make said switch.
--As these two rules are not used in and interfer with the benchmarking, they are commented out.
-- {-# RULES
-- "append -> fusedCoCh" [~1] forall t1 t2.
--   append t1 t2 =
--     fromCoCh (appendCoCh (toCoCh t1) (toCoCh t2))

-- "append -> unfusedCoCh" [1] forall t1 t2.
--   fromCoCh (appendCoCh (toCoCh t1) (toCoCh t2)) =
--     append t1 t2
-- #-}
