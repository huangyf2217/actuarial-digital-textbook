/* Chap4 SAS代码 */
/* 自动从chap4.html同步生成 */

/* 4.5.1 B-F法示例1 */
proc iml;
  /* 累积赔款流量三角形 */
  C = {2866 3334 3503 3624 3719 3720,
       3359 3889 4033 4231 4319 .,
       3848 4503 4779 4946 . .,
       4673 5422 5676 . . .,
       5369 6142 . . . .,
       5818 . . . . .};
  /* 已赚保费 */
  EP = {4486, 5024, 5680, 6590, 7482, 8502};
  /* 预期赔付率 */
  ELR = 0.83;

  n = nrow(C);
  /* 链梯进展因子 */
  f = j(n-1, 1, 0);
  do j = 1 to n-1;
    s1 = 0; s2 = 0;
    do i = 1 to n-j;
      s1 = s1 + C[i, j+1];
      s2 = s2 + C[i, j];
    end;
    f[j] = s1 / s2;
  end;

  /* 累积进展因子 */
  F = j(n, 1, 1);
  do i = 2 to n;
    F[i] = F[i-1] * f[n-i+1];
  end;

  /* B-F法准备金 */
  reserve = j(n, 1, 0);
  do i = 2 to n;
    ult_loss = EP[i] * ELR;
    reserve[i] = ult_loss * (1 - 1/F[i]);
  end;
  total_reserve = sum(reserve);
  print "B-F法各事故年准备金" reserve;
  print "总准备金 =" total_reserve;
quit;

/* 4.5.3 B-F法示例2 */
proc iml;
  C = {473 620 690 715,
       512 660 750 .,
       611 700 . .,
       647 . . .};
  EP = {860, 940, 980, 1020};
  ELR = 0.85;

  n = nrow(C);
  /* 链梯进展因子 */
  f = j(n-1, 1, 0);
  do j = 1 to n-1;
    s1 = 0; s2 = 0;
    do i = 1 to n-j;
      s1 = s1 + C[i, j+1];
      s2 = s2 + C[i, j];
    end;
    f[j] = s1 / s2;
  end;

  /* 累积进展因子 */
  F = j(n, 1, 1);
  do i = 2 to n;
    F[i] = F[i-1] * f[n-i+1];
  end;

  /* B-F法准备金 */
  reserve = j(n, 1, 0);
  do i = 2 to n;
    ult_loss = EP[i] * ELR;
    reserve[i] = ult_loss * (1 - 1/F[i]);
  end;
  total_reserve = sum(reserve);
  print "累积进展因子" F;
  print "B-F法各事故年准备金" reserve;
  print "总准备金 =" total_reserve;
quit;