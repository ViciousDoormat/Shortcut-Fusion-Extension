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
  bgroup "append"  [ bench "normal"   $ nf (uncurry append_)   (between (1,10000), between (1,10000))
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

--A tree with zero or one value at the leafs, and any number of branches.
data MultiTree a = None | End a | Branch [MultiTree a] deriving (Generic, NFData, Show)

data MultiTree_ a b = None_ | End_ a | Branch_ [b]

newtype MultiTreeCh a = MultiTreeCh (forall b. (MultiTree_ a b -> b) -> b)

fold :: (MultiTree_ a b -> b) -> MultiTree a -> b
fold a None = a None_
fold a (End x) = a (End_ x)
fold a (Branch xs) = a (Branch_ $ map (fold a) xs)

toCh :: MultiTree a -> MultiTreeCh a
toCh t = MultiTreeCh (`fold` t)
{-# INLINE [0] toCh #-}

in' :: MultiTree_ a (MultiTree a) -> MultiTree a
in' None_ = None
in' (End_ x) = End x
in' (Branch_ xs) = Branch xs

fromCh :: MultiTreeCh a -> MultiTree a
fromCh (MultiTreeCh g) = g in'
{-# INLINE [0] fromCh #-}

{-# RULES "toCh/fromCh fusion" forall x. toCh (fromCh x) = x #-}

--Given (x,y) create a multi tree with all values from x to y inclusive, such that each node has at most 3 children.
betweenCh :: (Int,Int) -> MultiTreeCh Int
betweenCh (x,y) = MultiTreeCh (`loop` (x,y))
  where
    loop :: (MultiTree_ Int b1 -> b1) -> (Int, Int) -> b1
    loop a (x,y) | x > y  = a None_
    loop a (x,y) | y - x + 1 <= 3 = case y - x + 1 of 1 -> a $ Branch_ [a (End_ x)]
                                                      2 -> a $ Branch_ [a (End_ x), a (End_ (x+1))]
                                                      3 -> a $ Branch_ [a (End_ x), a (End_ (x+1)), a (End_ (x+2))]
    loop a (x,y) = let spread = (y - x) `div` 3; toAdd = x+spread
                   in a (Branch_ [loop a (x, toAdd), loop a (toAdd+1, toAdd+1+spread), loop a (toAdd+2+spread, y)])

between' :: (Int,Int) -> MultiTree Int
between' = fromCh . betweenCh
{-# INLINE between' #-}

f :: (a -> Bool) -> MultiTree_ a b -> MultiTree_ a b
f p None_    = None_
f p (End_ x) = if p x then End_ x else None_
f p (Branch_ xs) = Branch_ xs

filterCh :: (a -> Bool) -> MultiTreeCh a -> MultiTreeCh a
filterCh p (MultiTreeCh g) = MultiTreeCh (\a -> g (a . f p))

filter' :: (a -> Bool) -> MultiTree a -> MultiTree a
filter' p = fromCh . filterCh p . toCh
{-# INLINE filter' #-}

r :: MultiTree_ a b -> MultiTree_ a b
r None_        = None_
r (End_ x)     = End_ x
r (Branch_ xs) = Branch_ (reverse xs)

reverseCh :: MultiTreeCh a -> MultiTreeCh a
reverseCh (MultiTreeCh g) = MultiTreeCh (\a -> g (a . r))

reverse' :: MultiTree a -> MultiTree a
reverse' = fromCh . reverseCh . toCh
{-# INLINE reverse' #-}

appendCh :: MultiTreeCh a -> MultiTreeCh a -> MultiTreeCh a
appendCh (MultiTreeCh g1) (MultiTreeCh g2) =
    MultiTreeCh (\a -> a (Branch_ [g1 a, g2 a]))

append' :: MultiTree a -> MultiTree a -> MultiTree a
append' t1 t2 = fromCh (appendCh (toCh t1) (toCh t2))
{-# INLINE append' #-}

mx :: MultiTree_ Int Int -> Int
mx None_        = minBound
mx (End_ x)     = x
mx (Branch_ xs) = maximum xs

maximumCh :: MultiTreeCh Int -> Int
maximumCh (MultiTreeCh g) = g mx

maximum' :: MultiTree Int -> Int
maximum' = maximumCh . toCh
{-# INLINE maximum' #-}

s :: MultiTree_ Int Int -> Int
s None_        = 0
s (End_ x)     = x
s (Branch_ xs) = sum xs

sumCh :: MultiTreeCh Int -> Int
sumCh (MultiTreeCh g) = g s

sum' :: MultiTree Int -> Int
sum' = sumCh . toCh
{-# INLINE sum' #-}

m :: (a -> b) -> MultiTree_ a c -> MultiTree_ b c
m f None_        = None_
m f (End_ x)     = End_ $ f x
m f (Branch_ xs) = Branch_ xs

mapCh :: (a -> b) -> MultiTreeCh a -> MultiTreeCh b
mapCh f (MultiTreeCh g) = MultiTreeCh (\a -> g (a . m f))

map' :: (a -> b) -> MultiTree a -> MultiTree b
map' f = fromCh . mapCh f . toCh
{-# INLINE map' #-}

sumApp' :: (Int, Int) -> Int
sumApp' (x,y) = sum' $ append' (between' (x,y)) (between' (x,y))

data MultiTreeCoCh a = forall s. MultiTreeCoCh (s -> MultiTree_ a s) s

out :: MultiTree a -> MultiTree_ a (MultiTree a)
out None = None_
out (End a) = End_ a
out (Branch xs) = Branch_ xs

toCoCh :: MultiTree a -> MultiTreeCoCh a
toCoCh = MultiTreeCoCh out
{-# INLINE [0] toCoCh #-}

unfold :: (s -> MultiTree_ a s) -> s -> MultiTree a
unfold h s = case h s of
  None_ -> None
  End_ a -> End a
  Branch_ xs -> Branch $ map (unfold h) xs

fromCoCh :: MultiTreeCoCh a -> MultiTree a
fromCoCh (MultiTreeCoCh h s) = unfold h s
{-# INLINE [0] fromCoCh #-}

{-# RULES "toCoCh/fromCoCh fusion" forall x. toCoCh (fromCoCh x) = x #-}

--Given (x,y) create a multi tree with all values from x to y inclusive, such that each node has at most 3 children.
betweenCoCh :: (Int,Int) -> MultiTreeCoCh Int
betweenCoCh (x,y) = MultiTreeCoCh h (x,y)
  where
    h :: (Int, Int) -> MultiTree_ Int (Int, Int)
    h (x,y) | x > y  = None_
            | y - x + 1 <= 3 = case y - x + 1 of 1 -> End_ x
                                                 2 -> Branch_ [(x, x), (x+1,x+1)]
                                                 3 -> Branch_ [(x,x), (x+1,x+1), (x+2, x+2)]
    h (x,y) = let spread = (y - x) `div` 3; toAdd = x+spread
              in Branch_ [(x,toAdd), (toAdd+1, toAdd+1+spread), (toAdd+2+spread, y)]

between'' :: (Int,Int) -> MultiTree Int
between'' = fromCoCh . betweenCoCh
{-# INLINE between'' #-}

maximumCoCh :: MultiTreeCoCh Int -> Int
maximumCoCh (MultiTreeCoCh h s) = loop s
  where
    loop s = case h s of
      None_    -> minBound
      End_ x   -> x
      Branch_ xs -> maximum $ map loop xs

maximum'' :: MultiTree Int -> Int
maximum'' = maximumCoCh . toCoCh
{-# INLINE maximum'' #-}

sumCoCh :: MultiTreeCoCh Int -> Int
sumCoCh (MultiTreeCoCh h s) = loop s
  where
    loop s = case h s of
      None_    -> 0
      End_ x   -> x
      Branch_ xs -> sum $ map loop xs

sum'' :: MultiTree Int -> Int
sum'' = sumCoCh . toCoCh
{-# INLINE sum'' #-}

reverseCoCh :: MultiTreeCoCh a -> MultiTreeCoCh a
reverseCoCh (MultiTreeCoCh h s) = MultiTreeCoCh (r . h) s

reverse'' :: MultiTree a -> MultiTree a
reverse'' = fromCoCh . reverseCoCh . toCoCh
{-# INLINE reverse'' #-}

filterCoCh :: (a -> Bool) -> MultiTreeCoCh a -> MultiTreeCoCh a
filterCoCh p (MultiTreeCoCh h s) = MultiTreeCoCh (f p . h) s

filter'' :: (a -> Bool) -> MultiTree a -> MultiTree a
filter'' p = fromCoCh . filterCoCh p . toCoCh
{-# INLINE filter'' #-}

appendCoCh :: MultiTreeCoCh a -> MultiTreeCoCh a -> MultiTreeCoCh a
appendCoCh (MultiTreeCoCh h1 s1) (MultiTreeCoCh h2 s2) = MultiTreeCoCh h' Nothing
  where
    h' Nothing = Branch_ [Just (MultiTreeCoCh h1 s1), Just (MultiTreeCoCh h2 s2)]
    h' (Just (MultiTreeCoCh h s)) = case h s of
      None_ -> None_
      End_ a -> End_ a
      Branch_ xs -> Branch_ $ map (Just . MultiTreeCoCh h) xs

append'' :: MultiTree a -> MultiTree a -> MultiTree a
append'' t1 t2 = fromCoCh $ appendCoCh (toCoCh t1) (toCoCh t2)
{-# INLINE append'' #-}

mapCoCh :: (a -> b) -> MultiTreeCoCh a -> MultiTreeCoCh b
mapCoCh f (MultiTreeCoCh h s) = MultiTreeCoCh (m f . h) s

map'' :: (a -> b) -> MultiTree a -> MultiTree b
map'' f = fromCoCh . mapCoCh f . toCoCh
{-# INLINE map'' #-}

sumApp'' :: (Int, Int) -> Int
sumApp'' (x,y) = sum'' $ append'' (between'' (x,y)) (between'' (x,y))

--define normal functions, benchmark

between :: (Int, Int) -> MultiTree Int
between (x,y) | x > y  = None
between (x,y) | y - x + 1 <= 3 = case y - x + 1 of 1 -> End x
                                                   2 -> Branch [End x, End (x+1)]
                                                   3 -> Branch [End x, End (x+1), End (x+2)]
between (x,y) = let spread = (y - x) `div` 3; toAdd = x+spread
                in Branch [between (x, toAdd), between (toAdd+1, toAdd+1+spread), between (toAdd+2+spread, y)]

filter_ :: (a -> Bool) -> MultiTree a -> MultiTree a
filter_ p None    = None
filter_ p (End x) = if p x then End x else None
filter_ p (Branch xs) = Branch $ map (filter_ p) xs

reverse_ :: MultiTree a -> MultiTree a
reverse_ None        = None
reverse_ (End x)     = End x
reverse_ (Branch xs) = Branch (reverse $ map reverse_ xs)

append_ :: MultiTree a -> MultiTree a -> MultiTree a
append_ t1 t2 = Branch [t1, t2]

maximum_ :: MultiTree Int -> Int
maximum_ None        = minBound
maximum_ (End x)     = x
maximum_ (Branch xs) = maximum $ map maximum_ xs

sum_ :: MultiTree Int -> Int
sum_ None        = 0
sum_ (End x)     = x
sum_ (Branch xs) = sum $ map sum_ xs

map_ :: (a -> b) -> MultiTree a -> MultiTree b
map_ f None        = None
map_ f (End x)     = End $ f x
map_ f (Branch xs) = Branch $ map (map_ f) xs

sumApp :: (Int, Int) -> Int
sumApp (x,y) = sum_ $ append_ (between (x,y)) (between (x,y))