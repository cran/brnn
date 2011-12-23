\name{normalize}
\alias{normalize}
\title{normalize}
\description{
  This function makes a linear transformation of the inputs such that the  values lie between -1 and 1.
}
\usage{
   normalize(y)
}
\arguments{
  \item{y}{a vector or matrix that needs to be normalized.}
}

\details{
  y=2*(y-my_min)/(my_max-my_min)-1
}
\value{
  A vector o matrix with the resulting normalized values.
}



\keyword{models}
