#include <vector>

#include "gtest/gtest.h"

#include "ideals.h"
#include "polynomials.h"


void expect_probabilities_near(const std::vector<double>& actual,
                               const std::vector<double>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_NEAR(actual[i], expected[i], 1e-12);
}


TEST(cyclic, example) {
  Polynomial x = {{1, {1,0,0}}};
  Polynomial y = {{1, {0,1,0}}};
  Polynomial z = {{1, {0,0,1}}};
  Polynomial one = {{1, {0,0,0}}};
  std::vector<Polynomial> F = {x + y + z, x*y + y*z + z*x, x*y*z - one};
  EXPECT_EQ(cyclic(3), F);
}


TEST(basis, example1) {
  std::vector<Monomial> B = {{}};
  EXPECT_EQ(basis(3, 0), B);
}


TEST(basis, example2) {
  std::vector<Monomial> B = {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1}};
  EXPECT_EQ(basis(4, 1), B);
}


TEST(basis, example3) {
  std::vector<Monomial> B = {{2,0,0}, {1,1,0}, {1,0,1}, {0,2,0}, {0,1,1}, {0,0,2}};
  EXPECT_EQ(basis(3, 2), B);
}


TEST(basis, example4) {
  std::vector<Monomial> B = {{3,0,0}, {2,1,0}, {2,0,1}, {1,2,0}, {1,1,1}, {1,0,2}, {0,3,0}, {0,2,1}, {0,1,2}, {0,0,3}};
  EXPECT_EQ(basis(3, 3), B);
}


TEST(binomial, example) {
  EXPECT_EQ(binomial(5, 0), 1);
  EXPECT_EQ(binomial(5, 1), 5);
  EXPECT_EQ(binomial(5, 2), 10);
  EXPECT_EQ(binomial(5, 3), 10);
  EXPECT_EQ(binomial(5, 4), 5);
  EXPECT_EQ(binomial(5, 5), 1);
}


TEST(degree_distribution, example1) {
  auto degree_dist = degree_distribution(3, 1, DistributionType::Weighted, false);
  std::vector<double> D = {0.0, 1.0};
  expect_probabilities_near(degree_dist.probabilities(), D);
}


TEST(degree_distribution, example2) {
  auto degree_dist = degree_distribution(3, 1, DistributionType::Weighted, true);
  std::vector<double> D = {0.5, 0.5};
  expect_probabilities_near(degree_dist.probabilities(), D);
}


TEST(degree_distribution, example3) {
  auto degree_dist = degree_distribution(3, 1, DistributionType::Uniform, true);
  std::vector<double> D = {0.25, 0.75};
  expect_probabilities_near(degree_dist.probabilities(), D);
}


TEST(degree_distribution, example4) {
  auto degree_dist = degree_distribution(3, 5, DistributionType::Weighted, false);
  std::vector<double> D = {0.0, 0.2, 0.2, 0.2, 0.2, 0.2};
  expect_probabilities_near(degree_dist.probabilities(), D);
}


TEST(degree_distribution, example5) {
  auto degree_dist = degree_distribution(3, 5, DistributionType::Weighted, true);
  std::vector<double> D = {1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6};
  expect_probabilities_near(degree_dist.probabilities(), D);
}


TEST(degree_distribution, example6) {
  auto degree_dist = degree_distribution(3, 5, DistributionType::Uniform, true);
  std::vector<double> D = {1.0/56, 3.0/56, 6.0/56, 10.0/56, 15.0/56, 21.0/56};
  expect_probabilities_near(degree_dist.probabilities(), D);
}


TEST(degree_distribution, example7) {
  auto degree_dist = degree_distribution(3, 3, DistributionType::Maximum, true);
  std::vector<double> D = {0.5, 0.0, 0.0, 0.5};
  expect_probabilities_near(degree_dist.probabilities(), D);
}


TEST(degree_distribution, example8) {
  auto degree_dist = degree_distribution(3, 3, DistributionType::Maximum, false);
  std::vector<double> D = {0.0, 0.0, 0.0, 1.0};
  expect_probabilities_near(degree_dist.probabilities(), D);
}


TEST(degree_distribution, example9) {
  auto degree_dist = degree_distribution(3, 3, DistributionType::Uniform, false);
  std::vector<double> D = {0.0, 3.0/19, 6.0/19, 10.0/19};
  expect_probabilities_near(degree_dist.probabilities(), D);
}


TEST(degree_distribution, example10) {
  auto degree_dist = degree_distribution(3, 3, DistributionType::Weighted, false);
  std::vector<double> D = {0.0, 1.0/3, 1.0/3, 1.0/3};
  expect_probabilities_near(degree_dist.probabilities(), D);
}


TEST(RandomBinomialIdealGenerator, example1) {
  RandomBinomialIdealGenerator ideal_gen{3, 5, 5};
  RandomBinomialIdealGenerator repeated_gen{3, 5, 5};
  ideal_gen.seed(123);
  repeated_gen.seed(123);
  auto F = ideal_gen.next();
  EXPECT_EQ(F, repeated_gen.next());
  EXPECT_EQ(F.size(), 5);
  for (const auto& f : F)
    EXPECT_EQ(f.terms.size(), 2);
}


TEST(RandomIdealGenerator, example1) {
  RandomIdealGenerator ideal_gen{3, 5, 5, 0.5};
  RandomIdealGenerator repeated_gen{3, 5, 5, 0.5};
  ideal_gen.seed(123);
  repeated_gen.seed(123);
  auto F = ideal_gen.next();
  EXPECT_EQ(F, repeated_gen.next());
  EXPECT_EQ(F.size(), 5);
  for (const auto& f : F)
    EXPECT_GE(f.terms.size(), 1);
}
