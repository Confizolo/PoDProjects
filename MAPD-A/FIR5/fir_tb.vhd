LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;

USE STD.textio.ALL;
-------------------------------------------------------------------------------

ENTITY fir_filter_5_tb IS

END ENTITY fir_filter_5_tb;

-------------------------------------------------------------------------------
ARCHITECTURE test OF fir_filter_5_tb IS
  COMPONENT fir_filter_5 IS
    PORT (
      i_clk : IN STD_LOGIC;
      i_rstb : IN STD_LOGIC;
      -- coefficient
      i_coeff_0 : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
      i_coeff_1 : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
      i_coeff_2 : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
      i_coeff_3 : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
      i_coeff_4 : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
      -- data input
      i_data : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
      -- filtered data 
      o_data : OUT STD_LOGIC_VECTOR(7 DOWNTO 0));
  END COMPONENT;
  -- component ports
  SIGNAL i_clk : STD_LOGIC := '0'; -- [in]
  SIGNAL i_rstb : STD_LOGIC; -- [in]
  SIGNAL i_coeff_0 : STD_LOGIC_VECTOR(7 DOWNTO 0) := X"09"; -- [in]
  SIGNAL i_coeff_1 : STD_LOGIC_VECTOR(7 DOWNTO 0) := X"3D"; -- [in]
  SIGNAL i_coeff_2 : STD_LOGIC_VECTOR(7 DOWNTO 0) := X"72"; -- [in]
  SIGNAL i_coeff_3 : STD_LOGIC_VECTOR(7 DOWNTO 0) := X"3D"; -- [in]
  SIGNAL i_coeff_4 : STD_LOGIC_VECTOR(7 DOWNTO 0) := X"09"; -- [in]
  SIGNAL i_data : STD_LOGIC_VECTOR(7 DOWNTO 0) := X"B2"; -- [in]
  SIGNAL o_data : STD_LOGIC_VECTOR(7 DOWNTO 0); -- [out]
  SIGNAL clk_enable : BOOLEAN := true;
  CONSTANT c_WIDTH : NATURAL := 8;
  FILE file_VECTORS : text;
  FILE file_RESULTS : text;

  -- clock

BEGIN -- architecture test

  -- component instantiation
  DUT : fir_filter_5
  PORT MAP(
    i_clk => i_clk,
    i_rstb => i_rstb,
    i_coeff_0 => i_coeff_0,
    i_coeff_1 => i_coeff_1,
    i_coeff_2 => i_coeff_2,
    i_coeff_3 => i_coeff_3,
    i_coeff_4 => i_coeff_4,
    i_data => i_data,
    o_data => o_data);

  -- clock generation
  i_clk <= NOT i_clk AFTER 10 ns WHEN clk_enable = true
  ELSE
  '0';
  -- waveform generation
  WaveGen_Proc : PROCESS
    VARIABLE CurrentLine : line;
    VARIABLE v_ILINE : line;
    VARIABLE v_OLINE : line;
    VARIABLE i_data_integer : INTEGER := 0;
    VARIABLE o_data_integer : INTEGER := 0;
    VARIABLE i_data_slv : STD_LOGIC_VECTOR(7 DOWNTO 0);
  BEGIN
    -- insert signal assignments here
    file_open(file_VECTORS, "input_vectors.txt", read_mode);
    file_open(file_RESULTS, "output_results.txt", write_mode);
    i_rstb <= '0';
    WAIT UNTIL rising_edge(i_clk);
    WAIT UNTIL rising_edge(i_clk);
    i_rstb <= '1';
    WHILE NOT endfile(file_VECTORS) LOOP
      readline(file_VECTORS, v_ILINE);
      read(v_ILINE, i_data_integer);
      i_data <= STD_LOGIC_VECTOR(to_signed(i_data_integer, i_data'length));
      WAIT UNTIL rising_edge(i_clk);
      o_data_integer := to_integer(signed(o_data));
      write(v_OLINE, o_data_integer, left, c_WIDTH);
      writeline(file_RESULTS, v_OLINE);
    END LOOP;
    file_close(file_VECTORS);
    file_close(file_RESULTS);
    clk_enable <= false;
    WAIT;
  END PROCESS WaveGen_Proc;

END ARCHITECTURE test;