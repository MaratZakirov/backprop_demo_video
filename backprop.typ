#set page(paper: "a4", margin: 2.2cm)
#set text(font: "New Computer Modern", size: 11pt, lang: "ru")
#set par(justify: true)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

#align(center)[
  #text(size: 20pt, weight: "bold")[
    Linear Softmax 1D, линейный слой и свёртки
  ]
  #v(0.3em)
  #text(size: 10pt)[
    Backpropagation-заметки: batch — первая размерность, embedding/feature — последняя
  ]
]

#v(0.5em)

= Обозначения

$X$ — вход, $Y$ — выход, $L$ — функция потерь.  
$partial$ — частная производная (∂), $delta$ — символ Кронекера (δ).  
Для softmax-части: $i,k$ — индексы batch; $j,l,p$ — индексы embedding/feature.  
Для свёрток: $n$ — batch, $c$ — канал, $y$ — позиция, $f$ — фильтр, $k$ — смещение ядра.  
Во всех формулах с повторяющимися индексами подразумевается суммирование, если оно не выписано явно.

= Простейший линейный слой

Forward с bias:

$ Y_(i,j) = X_(i,p) W_(p,j) + B_j $

Градиенты:

Для W

$ frac(partial L, partial W_(p,j)) = frac(partial L, partial Y_(i,j)) dot frac(partial Y_(i,j), partial W_(p,j)) $
$ frac(partial L, partial W_(p,j)) = frac(partial L, partial Y_(i,j)) dot frac(partial (X_(i,p') W_(p',j) + B_j), partial W_(p,j)) $
$ frac(partial L, partial W_(p,j)) = frac(partial L, partial Y_(i,j)) dot frac(partial W_(p',j), partial W_(p,j)) dot X_(i,p') $
$ frac(partial L, partial W_(p,j)) = frac(partial L, partial Y_(i,j)) dot delta_(p,p') dot X_(i,p') $
$ frac(partial L, partial W_(p,j)) = frac(partial L, partial Y_(i,j)) dot X_(i,p) $

Для B

$ frac(partial L, partial B_j) = frac(partial L, partial Y_(i,j)) dot frac(partial Y_(i,j), partial B_j) $
$ frac(partial L, partial B_j) = frac(partial L, partial Y_(i,j)) dot frac(partial (X_(i,p') W_(p',j) + B_j), partial B_j) $
$ frac(partial L, partial B_j) = frac(partial L, partial Y_(i,j)) dot frac(partial B_j, partial B_j) $
$ frac(partial L, partial B_j) = frac(partial L, partial Y_(i,j)) dot 1 $
$ frac(partial L, partial B_j) = frac(partial L, partial Y_(i,j)) $

Для X

$ frac(partial L, partial X_(i,p)) = frac(partial L, partial Y_(i,j)) dot frac(partial Y_(i,j), partial X_(i,p)) $
$ frac(partial L, partial X_(i,p)) = frac(partial L, partial Y_(i,j)) dot frac(partial (X_(i,p') W_(p',j) + B_j), partial X_(i,p)) $
$ frac(partial L, partial X_(i,p)) = frac(partial L, partial Y_(i,j)) dot frac(partial X_(i,p'), partial X_(i,p)) dot W_(p',j) $
$ frac(partial L, partial X_(i,p)) = frac(partial L, partial Y_(i,j)) dot delta_(p,p') dot W_(p',j) $
$ frac(partial L, partial X_(i,p)) = frac(partial L, partial Y_(i,j)) dot W_(p,j) $

#block(
  fill: rgb("#f7f7f7"),
  inset: 8pt,
  radius: 4pt,
  stroke: 0.5pt + rgb("#dddddd"),
)[
  *Комментарий.* Это самый простой линейный слой: $X$ имеет размерность `batch × in_features`, $W$ — `in_features × out_features`, $Y$ — `batch × out_features`. Во всех финальных строках подразумевается сумма по повторяющимся индексам ($i$ для $W$ и $B$, $j$ для $X$).
]

= Softmax по последней размерности

Forward:

$ Y_(i,j) = e^(X_(i,j)) / (sum_p e^(X_(i,p))) $

Дифференцируем по входу:

$ (partial L)/(partial X_(k,l)) = (partial L)/(partial Y_(i,j)) (partial Y_(i,j))/(partial X_(k,l)) $

$ (partial Y_(i,j))/(partial X_(k,l)) = partial/(partial X_(k,l)) (e^(X_(i,j)) / (sum_p e^(X_(i,p)))) $

По правилу частного:

$ (partial Y_(i,j))/(partial X_(k,l)) = 1/((sum_p e^(X_(i,p)))^2) ( (partial e^(X_(i,j)))/(partial X_(k,l)) dot sum_p e^(X_(i,p)) - e^(X_(i,j)) dot (partial (sum_p e^(X_(i,p))))/(partial X_(k,l)) ) $

$ (partial Y_(i,j))/(partial X_(k,l)) = 1/((sum_p e^(X_(i,p)))^2) ( e^(X_(i,j)) delta_(i k) delta_(j l) dot sum_p e^(X_(i,p)) - e^(X_(i,j)) dot e^(X_(i,l)) delta_(i k) ) $

Заменяем $i$ на $k$ там, где это позволяет $delta_(i k)$:

$ (partial Y_(i,j))/(partial X_(k,l)) = 1/((sum_p e^(X_(i,p)))^2) ( e^(X_(k,l)) delta_(i k) delta_(j l) dot sum_p e^(X_(i,p)) - e^(X_(k,j)) dot e^(X_(k,l)) delta_(i k) ) $

Ещё раз, с учётом $delta_(i k)$ в знаменателе:

$ (partial Y_(i,j))/(partial X_(k,l)) = 1/((sum_p e^(X_(k,p)))^2) ( e^(X_(k,l)) delta_(i k) delta_(j l) dot sum_p e^(X_(k,p)) - e^(X_(k,j)) dot e^(X_(k,l)) delta_(i k) ) $

Вносим знаменатель:

$ (partial Y_(i,j))/(partial X_(k,l)) = (e^(X_(k,l)) dot sum_p e^(X_(k,p)))/((sum_p e^(X_(k,p)))^2) delta_(i k) delta_(j l) - (e^(X_(k,j)) dot e^(X_(k,l)))/((sum_p e^(X_(k,p)))^2) delta_(i k) $

Упрощаем:

$ (partial Y_(i,j))/(partial X_(k,l)) = (e^(X_(k,l)))/(sum_p e^(X_(k,p))) delta_(i k) delta_(j l) - (e^(X_(k,j)) dot e^(X_(k,l)))/((sum_p e^(X_(k,p)))^2) delta_(i k) $

Ещё раз:

$ (partial Y_(i,j))/(partial X_(k,l)) = Y_(k,l) delta_(i k) delta_(j l) - Y_(k,j) Y_(k,l) delta_(i k) $

Подставляем в градиент потерь:

$ (partial L)/(partial X_(k,l)) = (partial L)/(partial Y_(i,j)) ( Y_(k,l) delta_(i k) delta_(j l) - Y_(k,j) Y_(k,l) delta_(i k) ) $

Сворачиваем $delta_(i k)$:

$ (partial L)/(partial X_(k,l)) = (partial L)/(partial Y_(k,j)) ( Y_(k,l) delta_(j l) - Y_(k,j) Y_(k,l) ) $

$ (partial L)/(partial X_(k,l)) = (partial L)/(partial Y_(k,l)) Y_(k,l) - (partial L)/(partial Y_(k,j)) Y_(k,j) Y_(k,l) $

Итог:

$ (partial L)/(partial X_(k,l)) = Y_(k,l) ( (partial L)/(partial Y_(k,l)) - sum_j (partial L)/(partial Y_(k,j)) Y_(k,j) ) $

#block(
  fill: rgb("#f7f7f7"),
  inset: 8pt,
  radius: 4pt,
  stroke: 0.5pt + rgb("#dddddd"),
)[
  *Замечание.* В последней строке сумма по $j$ восстановлена — без неё градиент softmax некорректен.
]

= Свёртка: градиент по весам

Forward-форма свёртки с явной суммой:

$ Y_(n,f,y) = sum_(c,k) X_(n,c,y+k) W_(f,c,k) $

Градиент по весам:

$ (partial L)/(partial W_(f,c,k)) = (partial L)/(partial Y_(n,f,y)) (partial Y_(n,f,y))/(partial W_(f,c,k)) $

$ (partial L)/(partial W_(f,c,k)) = (partial L)/(partial Y_(n,f,y)) partial/(partial W_(f,c,k)) (sum_(c',k') X_(n,c',y+k') W_(f,c',k')) $

$ (partial L)/(partial W_(f,c,k)) = (partial L)/(partial Y_(n,f,y)) (partial W_(f,c',k'))/(partial W_(f,c,k)) X_(n,c',y+k') $

$ (partial L)/(partial W_(f,c,k)) = (partial L)/(partial Y_(n,f',y)) (partial W_(f',c',k'))/(partial W_(f,c,k)) X_(n,c',y+k') $

$ (partial L)/(partial W_(f,c,k)) = (partial L)/(partial Y_(n,f',y)) delta_(f f') delta_(c c') delta_(k k') X_(n,c',y+k') $

$ (partial L)/(partial W_(f,c,k)) = (partial L)/(partial Y_(n,f,y)) X_(n,c,y+k) $

= Свёртка: градиент по входу

$ (partial L)/(partial X_(n,c,y)) = (partial L)/(partial Y_(n',f',y')) (partial Y_(n',f',y'))/(partial X_(n,c,y)) $

$ (partial L)/(partial X_(n,c,y)) = (partial L)/(partial Y_(n',f',y')) partial/(partial X_(n,c,y)) (sum_(c',k') X_(n',c',y'+k') W_(f',c',k')) $

$ (partial L)/(partial X_(n,c,y)) = (partial L)/(partial Y_(n',f',y')) (partial X_(n',c',y'+k'))/(partial X_(n,c,y)) W_(f',c',k') $

$ (partial L)/(partial X_(n,c,y)) = (partial L)/(partial Y_(n',f',y')) delta_(n n') delta_(c c') delta_(y'+k',y) W_(f',c',k') $

$ (partial L)/(partial X_(n,c,y)) = (partial L)/(partial Y_(n,f',y-k')) W_(f',c,k') $

#block(
  fill: rgb("#f7f7f7"),
  inset: 8pt,
  radius: 4pt,
  stroke: 0.5pt + rgb("#dddddd"),
)[
  *Комментарий.* В финальных формулах для свёрток подразумевается суммирование по повторяющимся индексам; в последней строке — также сумма по $f'$ и $k'$.
]
