from manim import *

class FFForwardPass(Scene):
    def construct(self):
        # 0. Заголовок наверху экрана <--- НОВОЕ
        title = Typst("*Forward Pass*", font_size=52, color=WHITE).to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1.5)

        # 1. Центральный блок (прямоугольник со скругленными углами)
        box = RoundedRectangle(
            corner_radius=0.2,
            width=3.2,
            height=1.8,
            color=BLUE,
            fill_opacity=0.15
        )
        box_label = MathTypst(r"X_(i, p) W_(p, j) + B_j", font_size=44).move_to(box.get_center())

        # 2. Стрелка слева (вход) -> упирается в левый край блока
        arrow_in = Arrow(
            start=box.get_left() + LEFT * 2.2,
            end=box.get_left(),
            buff=0.05,  # небольшой зазор до рамки
            color=WHITE
        )
        label_in = MathTypst(r"X_(i,p)", font_size=36).next_to(arrow_in, UP, buff=0.15)

        # 3. Стрелка справа (выход) -> выходит из правого края блока
        arrow_out = Arrow(
            start=box.get_right(),
            end=box.get_right() + RIGHT * 2.2,
            buff=0.05,
            color=GREEN
        )
        label_out = MathTypst(r"Y_(i,j)", font_size=36, color=GREEN).next_to(arrow_out, UP, buff=0.15)

        # 4. Стрелка снизу (управление / измерения) -> входит в нижний край блока
        arrow_bottom = Arrow(
            start=box.get_bottom() + DOWN * 1.8,
            end=box.get_bottom(),
            buff=0.05,
            color=YELLOW
        )
        label_bottom = MathTypst(r"W_(p,j)", font_size=36, color=YELLOW).next_to(arrow_bottom, RIGHT, buff=0.2)

        # 5. Плавная пошаговая анимация сборки схемы
        self.play(Create(box), Write(box_label), run_time=2.2)
        self.play(
            GrowArrow(arrow_in), Write(label_in),
            GrowArrow(arrow_bottom), Write(label_bottom),
            run_time=2.2
        )
        self.play(GrowArrow(arrow_out), Write(label_out), run_time=2.0)
        self.wait(15)

class FFBackwardPass(Scene):
    def construct(self):
        # 0. Заголовок наверху экрана
        title = Typst("*Backward Pass*", font_size=52, color=WHITE).to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1.5)

        # 1. Центральный блок (чуть расширили width=4.0, чтобы длинная формула не вылезала за рамки)
        box = RoundedRectangle(
            corner_radius=0.2,
            width=4.0,
            height=1.8,
            color=BLUE,
            fill_opacity=0.15
        )
        box_label = MathTypst(r"X_(i, p) W_(p, j) + B_j", font_size=40).move_to(box.get_center())

        # 2. Стрелка справа (входной градиент сверху) -> теперь ВХОДИТ в правый край блока
        arrow_right = Arrow(
            start=box.get_right() + RIGHT * 2.2,
            end=box.get_right(),
            buff=0.05,
            color=WHITE
        )
        label_right = MathTypst(r"frac(partial L, partial Y_(i,j))", font_size=34, color=WHITE).next_to(arrow_right, UP, buff=0.15)

        # 3. Стрелка слева (градиент по входу X) -> теперь ВЫХОДИТ из левого края блока
        arrow_left = Arrow(
            start=box.get_left(),
            end=box.get_left() + LEFT * 2.2,
            buff=0.05,
            color=GREEN
        )
        label_left = MathTypst(r"frac(partial L, partial X_(i,p))", font_size=34, color=GREEN).next_to(arrow_left, UP, buff=0.15)

        # 4. Стрелка снизу (градиент по весам W) -> теперь ВЫХОДИТ из нижнего края блока вниз
        arrow_bottom = Arrow(
            start=box.get_bottom(),
            end=box.get_bottom() + DOWN * 1.8,
            buff=0.05,
            color=YELLOW
        )
        label_bottom = MathTypst(r"frac(partial L, partial W_(p,j))", font_size=34, color=YELLOW).next_to(arrow_bottom, RIGHT, buff=0.2)

        # 5. Плавная сборка по ходу обратного распространения ошибки (справа -> налево и вниз)
        self.play(Create(box), Write(box_label), run_time=2.2)
        self.play(GrowArrow(arrow_right), Write(label_right), run_time=2.0)
        self.play(
            GrowArrow(arrow_left), Write(label_left),
            GrowArrow(arrow_bottom), Write(label_bottom),
            run_time=2.2
        )
        self.wait(15)

class LinearLayerDerivation(Scene):
  def animate_derivation_block(
      self, section_title: str, steps: list[str], font_size: int = 38
  ):
    """Плавно выводит заголовок раздела и пошагово трансформирует формулы,

    оставляя историю вычислений сверху в полупрозрачном виде.
    """
    header = Typst(section_title, font_size=44, color=BLUE_B).to_edge(UP, buff=0.4)

    self.play(Write(header), run_time=0.7)

    # Отображаем первый шаг по центру экрана
    current_eq = MathTypst(steps[0], font_size=font_size).move_to(ORIGIN)
    self.play(Write(current_eq), run_time=1.5)
    self.wait(1.0)

    history = VGroup()

    for next_step_tex in steps[1:]:
      # Копируем текущую формулу и отправляем её в историю наверх
      prev_copy = current_eq.copy()
      history.add(prev_copy)

      # Сдвигаем всю историю выше и делаем её полупрозрачной для фокуса на новом шаге
      target_y = 2.2 - 1.05 * (len(history) - 1)
      self.play(
          prev_copy.animate.scale(0.75)
          .set_opacity(0.65)
          .move_to(UP * target_y),
          run_time=1.5,
      )

      # Плавно трансформируем центральную формулу в следующий шаг вывода
      next_eq = MathTypst(next_step_tex, font_size=font_size).move_to(
          DOWN * 1.2
      )
      self.play(ReplacementTransform(current_eq, next_eq), run_time=1.1)
      current_eq = next_eq
      self.wait(5.5)

    # Подсвечиваем финальную формулу рамкой
    final_box = SurroundingRectangle(
        current_eq, color=GREEN, buff=0.2, corner_radius=0.1
    )
    self.play(Create(final_box), run_time=0.8)
    self.wait(10.0)

    # Очищаем экран перед следующим разделом
    self.play(FadeOut(VGroup(header, history, current_eq, final_box)), run_time=0.7)

  def construct(self):
    # --- Блок 1: Прямой проход ---
    forward_steps = [
        r"Y_(i, j) = X_(i, p) W_(p, j) + B_j",
    ]

    self.animate_derivation_block(
        "Forward Pass", forward_steps, font_size=46
    )

    # --- Блок 2: Градиент по входу dL/dX ---
    dx_steps = [
        r"frac(partial L, partial X_(k, l)) = frac(partial L, partial Y_(i, j)) frac(partial Y_(i, j), partial X_(k, l))",
        r"frac(partial L, partial X_(k, l)) = frac(partial L, partial Y_(i, j)) frac(partial (X_(i, p) W_(p, j) + B_j), partial X_(k, l))",
        r"frac(partial L, partial X_(k, l)) = frac(partial L, partial Y_(i, j)) delta_(i, k) delta_(p, l) W_(p, j)",
        r"frac(partial L, partial X_(k, l)) = frac(partial L, partial Y_(k, j)) W_(l, j)",
    ]
    self.animate_derivation_block(
        r"*Deriving:* $(partial L) / (partial X)$",
        dx_steps,
        font_size=38,
    )

    # --- Блок 3: Градиент по весам dL/dW ---
    dw_steps = [
        r"frac(partial L, partial W_(k, l)) = frac(partial L, partial Y_(i, j)) frac(partial Y_(i, j), partial W_(k, l))",
        r"frac(partial L, partial W_(k, l)) = frac(partial L, partial Y_(i, j)) frac(partial (X_(i, p) W_(p, j) + B_j), partial W_(k, l))",
        r"frac(partial L, partial W_(k, l)) = frac(partial L, partial Y_(i, j)) X_(i, p) delta_(p, k) delta_(j, l)",
        r"frac(partial L, partial W_(k, l)) = X_(i, k) frac(partial L, partial Y_(i, l))",
    ]
    self.animate_derivation_block(
        r"*Deriving:* $(partial L) / (partial W)$",
        dw_steps,
        font_size=38,
    )

if __name__ == "__main__":
  cfg = {"quality": "high_quality", "preview": False}

  with tempconfig(cfg):
      FFForwardPass().render()

  with tempconfig(cfg):
      FFBackwardPass().render()

  with tempconfig(cfg):
      LinearLayerDerivation().render()